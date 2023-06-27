from abc import ABC, abstractmethod

import activelearning as al
import numpy as np
import tensorflow as tf
import tensorflow.keras as K


def tf_expected_information_gain(tf_x, learner, batch_size):
    """
    Information gain loss function
    """
    tf_samples_lambda = learner.prior.sample(batch_size)
    tf_H_lambda = -tf.reduce_mean(learner.prior.log_prob(tf_samples_lambda))

    tf_samples_y_given_lambda = learner.likelihood.sample(
        1, tf_x=tf_x, tf_lambda=tf_samples_lambda
    )[0]

    # Log prob could be sampled on non-zero probability samples
    # to avoid numerical instabilities!
    # tf_samples_lambda_posterior = tf.stop_gradient(
    #     learner.posterior.sample(batch_size,conditional_arguments=tf_samples_y_given_lambda))
    # However, this is not numerically ok when taking the gradient.
    tf_log_q = learner.posterior.log_prob(
        tf_samples_lambda, conditional_arguments=tf_samples_y_given_lambda
    )

    tf_H_lambda_given_y = -tf.reduce_mean(tf_log_q)
    tf_IG_x = tf_H_lambda - tf_H_lambda_given_y
    return tf_IG_x


def tf_expected_information_gain_qubits(tf_x, learner, batch_size):
    # We need to use probs for y instead of sampling because
    # we cannot use the reparametrization trick
    tf_samples_lambda = learner.prior.sample(batch_size)
    tf_H_lambda = -tf.reduce_mean(learner.prior.log_prob(tf_samples_lambda))

    tf_samples_y_given_lambda_1 = tf.exp(
        learner.likelihood.log_prob(1, tf_x=tf_x, tf_lambda=tf_samples_lambda)
    )
    tf_samples_y_given_lambda_m1 = tf.exp(
        learner.likelihood.log_prob(0, tf_x=tf_x, tf_lambda=tf_samples_lambda)
    )

    tf_log_q_1 = learner.posterior.log_prob(
        tf_samples_lambda, conditional_arguments=tf.ones([batch_size, 1])
    )
    tf_log_q_m1 = learner.posterior.log_prob(
        tf_samples_lambda, conditional_arguments=tf.zeros([batch_size, 1])
    )

    tf_H_lambda_given_y = -tf.reduce_mean(
        tf_log_q_1 * tf_samples_y_given_lambda_1
        + tf_log_q_m1 * tf_samples_y_given_lambda_m1
    )

    # Ignoring outliers
    weighted_sum = (
        tf_log_q_1 * tf_samples_y_given_lambda_1
        + tf_log_q_m1 * tf_samples_y_given_lambda_m1
    )
    allowed_samples = tf.cast(weighted_sum > -10.0, K.backend.floatx())
    n_allowed = tf.reduce_sum(allowed_samples)
    tf_H_lambda_given_y = -tf.reduce_sum(weighted_sum * allowed_samples) / n_allowed

    tf_IG_x = tf_H_lambda - tf_H_lambda_given_y

    return tf_IG_x


def tf_expected_information_gain_qubits_binomial(tf_x, learner, batch_size):
    ### H(λ) = Σ logPrior(λ)
    tf_samples_lambda = learner.prior.sample(batch_size)
    tf_H_lambda = -tf.reduce_mean(learner.prior.log_prob(tf_samples_lambda))

    ### H(λ|y;x) = ΣΣ Prior(λ) Likelihood(y|λ;x) log Posterior(λ|y;x)
    y_values = (
        tf.range(learner.system.n_counts + 1, dtype=K.backend.floatx())
        / learner.system.n_counts
    )

    # shape [batch_size, y]
    tf_y_weights = learner.likelihood.prob(
        y_values, tf_x=tf_x, tf_lambda=tf_samples_lambda
    )

    def tf_log_q_given_y(tf_y):
        conditional_args = tf.repeat(tf_y[None], tf_samples_lambda.shape[0], axis=0)
        return learner.posterior.log_prob(
            tf_samples_lambda, conditional_arguments=conditional_args
        )

    # Calculates logQ(λ|y) at fixed x
    tf_log_q = tf.vectorized_map(tf_log_q_given_y, y_values[:, None])
    tf_log_q = tf.transpose(tf_log_q)  # shape [batch_size, y]
    # tf_H_lambda_given_y = - tf.reduce_mean(tf.reduce_sum(tf_log_q*tf_y_weights,1), 0)

    # Ignoring outliers
    weighted_sum = tf.reduce_sum(tf_log_q * tf_y_weights, 1)
    allowed_samples = tf.cast(weighted_sum > -5.0, K.backend.floatx())
    n_allowed = tf.reduce_sum(allowed_samples)
    tf_H_lambda_given_y = -tf.reduce_sum(weighted_sum * allowed_samples) / n_allowed

    tf_IG_x = tf_H_lambda - tf_H_lambda_given_y
    return tf_IG_x


class Metric:
    """
    A metric object helps tracking quantities to analyze from the experiments.
    """

    def __init__(self, history=None):
        if history is None:
            history = []
        self.history = history

    # @abstractmethod
    def evaluate(self, result, **kwargs):
        self.history.append(result)
        return self.history[-1]

    def __getitem__(self, str):
        return self.history[str]

    def save(self, h5_handle, metric_name):
        h5_handle.create_dataset(metric_name, data=self.history)

    def load(self, h5_handle):
        self.history = list(h5_handle[:])

    def recompile_tf_graph(self):
        ...


class ExpectedInformationGainCurve(Metric):
    def __init__(self, parallelAdvisor, x_range, n_train=3000):
        super().__init__()
        self.advisor = parallelAdvisor
        self.x_range = x_range
        self.n_train = n_train
        # TODO: maybe metric settings should be saved?
        # TODO: maybe we should be able to LOAD them back?

    def evaluate(self, **kwargs):
        al.parallel.init_active_learner(
            self.advisor.parallel, self.advisor.last_save_path
        )

        if self.advisor.parallel is None:
            raise Exception(
                "We need to use multiple nodes to be able to estimate it in a reasonable amount of time"
            )

        xs = np.linspace(*self.x_range, len(self.advisor.parallel))
        self.advisor.parallel.dview.scatter("xs", xs, block=True)
        self.advisor.parallel.execute("advisor.tf_x_optimal.assign(xs[0])")
        self.advisor.parallel.assign_task(
            f"advisor._train_posterior(n_train={self.n_train}, train_x=False)",
            show_progress=True,
            title="Estimating expected information gain curve...",
        )

        result_ig = self.advisor.parallel[
            "learner.metrics[al.metrics.ExpectedInformationGain.__name__].evaluate()"
        ]

        result = np.stack([xs, result_ig], -1)
        return super().evaluate(np.array(result))


class DiscreteExpectedInformationGainCurve(Metric):
    def __init__(self, discreteLearner, x_span):
        super().__init__()
        self.discreteLearner = discreteLearner
        self.x_span = x_span

    def evaluate(self, **kwargs):
        ig = self.discreteLearner.IG()
        result = np.stack([self.x_span, ig], -1)
        return super().evaluate(result)


class ParallelWorkersTraining(Metric):
    def __init__(self, parallel):
        super().__init__()
        self.parallel = parallel

    def evaluate(self, **kwargs):
        result = np.stack(
            [
                np.array(self.parallel["advisor.training_history.information_gains"]).T,
                np.array(self.parallel["advisor.training_history.optimal_xs"]).T,
            ],
            -1,
        )

        return super().evaluate(result)


class Covariance(Metric):
    def __init__(self, learner, batch_size=100_000):
        super().__init__()
        self.learner = learner
        self.batch_size = batch_size

        self.recompile_tf_graph()

    def evaluate(self, **kwargs):
        return super().evaluate(self.tf_evaluate().numpy())

    def _tf_evaluate(self):
        tf_samples_lambda = self.learner.prior.sample(self.batch_size)

        tf_samples_lambda_mean = tf.reduce_mean(tf_samples_lambda, 0)
        return tf.reduce_mean(
            (tf_samples_lambda[:, None, :] - tf_samples_lambda_mean)
            * (tf_samples_lambda[:, :, None] - tf_samples_lambda_mean),
            axis=0,
        )

    def recompile_tf_graph(self):
        self.tf_evaluate = tf.function(self._tf_evaluate)


class OutputKullbackLeibler(Metric):
    def __init__(self, learner, x_range, n_points=50, batch_size=5000):
        super().__init__()
        self.learner = learner
        self.batch_size = batch_size
        self.x_range = x_range
        self.n_points = n_points

        self.recompile_tf_graph()

    def evaluate(self, **kwargs):
        x_vals = np.linspace(*self.x_range, self.n_points)
        result = [
            self.tf_evaluate_x(tf.convert_to_tensor(x, K.backend.floatx())).numpy()
            for x in x_vals
        ]
        return super().evaluate(result)

    # Todo: parallelize?
    # TF.FUNCTION DOES NOT WORK HERE?
    def _evaluate_x(self, tf_x):
        samples_y = self.learner.likelihood.sample(
            self.batch_size,
            tf_x=tf_x,
            tf_lambda=self.learner.system.tf_real_lambda[None, :],
        )
        tf_log_P = self.learner.likelihood.log_prob(
            samples_y, tf_x=tf_x, tf_lambda=self.learner.system.tf_real_lambda[None, :]
        )
        tf_samples_lambda = self.learner.prior.sample(self.batch_size * 2)
        tf_log_Q = tf.reduce_mean(
            self.learner.likelihood.log_prob(
                samples_y[:, None], tf_x=tf_x, tf_lambda=tf_samples_lambda
            ),
            axis=-1,
        )
        return tf.reduce_mean(tf_log_P - tf_log_Q, 0)

    def recompile_tf_graph(self):
        self.tf_evaluate_x = tf.function(self._evaluate_x)


class InformationGained(Metric):
    def __init__(self, advisor, batch_size=5000):
        super().__init__()
        self.advisor = advisor
        self.batch_size = batch_size

    def evaluate(self, tf_new_y_measure, **kwargs):
        result = self.get_information_gained_if_y(tf_new_y_measure).numpy()
        return super().evaluate(result)
        # measure at y should have shape [1,dim_y]

    def get_information_gained_if_y(self, tf_y):
        tf_samples_lambda = self.advisor.learner.prior.sample(self.batch_size)
        tf_H_lambda = -tf.reduce_mean(
            self.advisor.learner.prior.log_prob(tf_samples_lambda)
        )

        tf_q_samples = self.advisor.learner.posterior.sample(
            self.batch_size, conditional_arguments=tf_y
        )
        tf_H_lambda_new = -tf.reduce_mean(
            self.advisor.learner.posterior.log_prob(
                tf_q_samples, conditional_arguments=tf_y
            )
        )

        return tf_H_lambda - tf_H_lambda_new


class ExpectedInformationGain(Metric):
    def __init__(self, advisor, batch_size=10000):
        super().__init__()
        self.advisor = advisor
        self.batch_size = batch_size

    def evaluate(self, **kwargs):
        result = self.advisor.information_gain_function(
            tf_x=self.advisor.tf_x_optimal,
            learner=self.advisor.learner,
            batch_size=self.batch_size,
        )
        return super().evaluate(result.numpy())


class MetricList:
    def __init__(self):
        self._metrics = {}

    def __getitem__(self, str):
        return self._metrics[str]

    def append(self, metric):
        self._metrics[type(metric).__name__] = metric

    def items(self):
        return self._metrics.items()

    def keys(self):
        return self._metrics.keys()

    def evaluate(self, **kwargs):
        for _, metric in self.items():
            metric.evaluate(**kwargs)

    def save(self, h5_handle):
        for metric_name, metric in self.items():
            metric.save(h5_handle, metric_name)

    def recompile_tf_graphs(self, **kwargs):
        for _, metric in self.items():
            metric.recompile_tf_graph()

    def load_all(self, h5_handle):
        for k, v in h5_handle.items():
            if k in self.keys():
                self[k].load(h5_handle[k])
            else:
                self._metrics[k] = Metric(
                    list(v[:])
                )  # type('Metric', (object,), {'history': list(v[:])})

    def __repr__(self) -> str:
        val = ""
        for k in self.keys():
            val += f"- {k}\n"
        return f"""{self.__class__}
{val}"""
