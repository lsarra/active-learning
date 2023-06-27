from abc import ABC, abstractmethod
from dataclasses import dataclass

import activelearning as al
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tqdm import trange


class Advisor(ABC):
    @abstractmethod
    def suggest_next_measurement(self):
        ...


class InformationGainAdvisor(Advisor):
    """Implements the Active measurement strategy."""

    @dataclass
    class TrainingHistory:
        information_gains: np.ndarray = np.empty(shape=(0, 1))
        optimal_xs: np.ndarray = np.empty(shape=(0, 1))

        def append(self, loss, x):
            self.information_gains = np.append(self.information_gains, loss)
            self.optimal_xs = np.append(self.optimal_xs, x)

        def save(self, h5_handle):
            return al.utils.h5_save_dataclass(self, h5_handle)

        def load(self, h5_handle):
            return al.utils.h5_load_dataclass(self, h5_handle)

        def reset(self):
            self.information_gains = np.empty(shape=(0, 1))
            self.optimal_xs = np.empty(shape=(0, 1))

        def __len__(self) -> int:
            return len(self.optimal_xs)

    def __init__(
        self,
        learner,
        starting_x,
        batch_size=500,
        n_train=5000,
        train_x=True,
        parallel=None,
    ):
        self.learner = learner

        self.learner.advisor = (
            self  # Todo: hacky solution, the architecture can be improved
        )

        self.batch_size = batch_size
        self.n_train = n_train
        self.train_x = train_x
        self.parallel = parallel

        self.learner.metrics.append(al.metrics.InformationGained(self))
        self.learner.metrics.append(al.metrics.ExpectedInformationGain(self))

        if self.parallel is not None:
            # self.learner.metrics.append(al.metrics.ParallelWorkersTraining(self.parallel))  # Takes huge space!
            self.learner.metrics.append(
                al.metrics.ExpectedInformationGainCurve(
                    self,
                    x_range=self.learner.system.x_range,
                    n_train=al.config.cfg.METRICS.EXP_IG.N_TRAIN,
                )
            )

        # Training attributes
        self.tf_x_optimal = tf.Variable(
            starting_x, dtype=K.backend.floatx(), name="tf_x_optimal"
        )

        if type(self.learner.system).__name__ == al.systems.Qubits.__name__:
            self.information_gain_function = (
                al.metrics.tf_expected_information_gain_qubits
            )
        elif type(self.learner.system).__name__ == al.systems.QubitsBinomial.__name__:
            self.information_gain_function = (
                al.metrics.tf_expected_information_gain_qubits_binomial
            )
        else:
            self.information_gain_function = al.metrics.tf_expected_information_gain

        # TODO: add learning rate as option
        self.x_optimizer = tf.optimizers.Adam()
        self.posterior_optimizer = tf.optimizers.Adam()

        self.training_history = self.TrainingHistory([], [])

        self._last_save_path = None
        self.recompile_tf_graph()
        self.learner.metrics.recompile_tf_graphs()

    @classmethod
    def from_h5_handle(cls, learner, h5_handle, parallel=None):
        starting_x = h5_handle.attrs["x_optimal"]
        batch_size = h5_handle.attrs["batch_size"]
        n_train = h5_handle.attrs["n_train"]
        # train_x = h5_handle.attrs["train_x"]

        advisor = cls(
            learner=learner,
            starting_x=starting_x,
            batch_size=batch_size,
            n_train=n_train,
            parallel=parallel,
        )

        advisor.training_history.load(h5_handle["training_history"])
        return advisor

    @property
    def last_save_path(self):
        if self._last_save_path is None:
            raise Exception("Model has never been saved")
        return self._last_save_path

    @last_save_path.setter
    def last_save_path(self, val):
        self._last_save_path = val

    def train_posterior(self, *args, **kwargs):
        if self.parallel is not None:
            # TODO: does not make sense if x_train is false,
            # because parallelization is useful only to avoid getting stuck in local minima
            # when optimizing the 1-d x (or for plotting the information gain curve)
            return self._train_posterior_parallelized(*args, **kwargs)
        else:
            return self._train_posterior(*args, **kwargs)

    def tf_expected_information_gain(self, tf_x):
        return self.information_gain_function(
            tf_x=tf_x, learner=self.learner, batch_size=self.batch_size
        )

    def _train_step(self, tf_x, train_x):
        with tf.GradientTape() as tape:
            tape.watch(tf_x)
            tf_IG_x = self.tf_expected_information_gain(tf_x=tf_x)
            tf_loss = tf.debugging.check_numerics(-tf_IG_x, "is the prob too small?")

        variables = (*self.learner.posterior.trainable_variables, tf_x)
        grads = tape.gradient(tf_loss, variables)

        tf_grads_x = tf.debugging.check_numerics(grads[-1], "grads_x")  # careful
        tf_grads_posterior = [
            tf.debugging.check_numerics(g, "grads") for g in grads[:-1]
        ]

        tf_grads_posterior = [
            tf.clip_by_value(grad, -0.5, 0.5) for grad in tf_grads_posterior
        ]

        self.posterior_optimizer.apply_gradients(
            zip(tf_grads_posterior, self.learner.posterior.trainable_variables)
        )
        if train_x:
            self.x_optimizer.apply_gradients(zip([tf_grads_x], [tf_x]))
        tf_x = tf.debugging.check_numerics(tf_x, "tf_x")

        return tf_IG_x  # , grads

    def _train_posterior(self, n_train=None, train_x=None):
        if n_train is None:
            n_train = self.n_train
        if train_x is None:
            train_x = self.train_x

        def train_step(train_idx):
            tf.debugging.assert_all_finite(self.tf_x_optimal, "tf_x")
            ig = self.tf_train_step(
                self.tf_x_optimal, tf.convert_to_tensor(train_x, tf.bool)
            )
            self.training_history.append(ig.numpy(), self.tf_x_optimal.numpy())

            # WORKAROUND FOR PARALLEL RUN
            al.parallel.log_progress(train_idx, n_train)

        for i in trange(n_train):
            train_step(i)

        # LATE STOPPING:
        # Keep training after the maximum until the optimal x converged
        # It allows to reduce the total number of training steps.
        # TODO: training precision and this setting itself should have a setting
        # TODO: add setting to enable it
        min_steps = 500
        if n_train >= min_steps and train_x:

            def train_gradient_history():
                return abs(
                    np.gradient(self.training_history.optimal_xs[-min_steps:]).sum()
                )

            while train_gradient_history() > 0.05:
                i += 1
                train_step(i)

        return self.tf_x_optimal

    def _train_posterior_parallelized(
        self, *args, **kwargs
    ):  # n_train, train_x will be ignored
        al.parallel.init_active_learner(self.parallel, self.last_save_path)

        xs = np.random.uniform(*self.learner.system.x_range, len(self.parallel))
        self.parallel.dview.scatter("xs", xs, block=True)
        self.parallel.execute("advisor.tf_x_optimal.assign(xs[0])")
        self.parallel.execute("advisor.training_history.reset()")
        self.parallel.assign_task(
            f"advisor.suggest_next_measurement()",
            show_progress=True,
            title=f"Training posterior #{len(self.learner.history)}",
        )

        results_ig = self.parallel["advisor.training_history.information_gains[-1]"]
        results_x = self.parallel["advisor.training_history.optimal_xs[-1]"]

        smartest_worker = int(np.argmax(results_ig))
        self.tf_x_optimal.assign(results_x[smartest_worker])
        self.learner.posterior.weights = self.parallel["learner.posterior.weights"][
            smartest_worker
        ]
        worker_history = self.parallel["advisor.training_history"][smartest_worker]
        self.training_history.append(
            worker_history.information_gains, worker_history.optimal_xs
        )

        return self.tf_x_optimal

    # TODO: Problems when subclassing, so I cannot remove train_posterior() (not clear)
    def suggest_next_measurement(self):
        return self.train_posterior()

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["batch_size"] = self.batch_size
        h5_handle.attrs["n_train"] = self.n_train
        h5_handle.attrs["train_x"] = self.train_x
        h5_handle.attrs["x_optimal"] = self.tf_x_optimal.numpy()

        self.training_history.save(h5_handle.create_group("training_history"))

    def recompile_tf_graph(self):
        self.tf_train_step = tf.function(self._train_step)


class RandomAdvisor(InformationGainAdvisor):
    """Random measurement selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train_x=False)
        # todo: should merge kwargs dict

    def suggest_next_measurement(self):
        starting_x = np.random.uniform(*self.learner.system.x_range)
        self.tf_x_optimal.assign(starting_x)

        return super().train_posterior()

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__


class FixedAdvisor(InformationGainAdvisor):
    """Always measures at the same given point."""

    def __init__(self, starting_x, *args, **kwargs):
        super().__init__(*args, **kwargs, starting_x=starting_x, train_x=False)

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__


class UniformGridAdvisor(InformationGainAdvisor):
    """Performs a measurement sweep from left to right."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train_x=False)
        # todo: should merge kwargs dict
        self.measurement_list = np.linspace(
            *self.learner.system.x_range, al.config.cfg.TRAINING.N_MEASUREMENTS
        )

    def suggest_next_measurement(self):
        starting_x = self.measurement_list[len(self.learner.history)]
        self.tf_x_optimal.assign(starting_x)

        return super().train_posterior()

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__


def get_advisor_from_name(name):
    if name == InformationGainAdvisor.__name__:
        return InformationGainAdvisor
    elif name == RandomAdvisor.__name__:
        return RandomAdvisor
    elif name == FixedAdvisor.__name__:
        return FixedAdvisor
    elif name == UniformGridAdvisor.__name__:
        return UniformGridAdvisor

    raise Exception("Invalid advisor name!")
