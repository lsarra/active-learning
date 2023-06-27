import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
from activelearning.normalizing_flows import (
    MaskedAutoregressiveFlowNew,
    NormalizingFlows,
)
from activelearning.systems import System
import h5py

from tqdm import trange
import logging

import scipy as sp
import scipy.special

import activelearning as al

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass


class Learner(ABC):
    @dataclass
    class History:
        x: np.ndarray
        y: np.ndarray

        def __init__(self, system: System):
            self.x = np.empty(shape=(0, system.dim_x))  # 0,1
            self.y = np.empty(shape=(0, system.dim_y))  # 0,2

        def append(self, x, y):
            if len(x.shape) == 0:
                x = x[None, ...]

            if len(y.shape) == 1:
                self.x = np.concatenate([self.x, x[None, ...]], 0)
                self.y = np.concatenate([self.y, y[None, ...]], 0)
            elif len(y.shape) == 2:
                self.x = np.concatenate([self.x, x], 0)
                self.y = np.concatenate([self.y, y], 0)
            else:
                raise Exception(
                    f"Measurement shape is not correct. Got {x.shape} instead of [None, 1] for x."
                )

        def __len__(self) -> int:
            return len(self.x)

        @property
        def last_y(self):
            if len(self.y) != 0:
                return self.y[-1]

            raise Exception("No measurement available!")

        def save(self, h5_handle):
            return al.utils.h5_save_dataclass(self, h5_handle)

        def load(self, h5_handle):
            return al.utils.h5_load_dataclass(self, h5_handle)

        def __repr__(self) -> str:
            dict_str = ""
            for key in self.__dict__.keys():
                dict_str += f"{key} "
            return f"""{self.__class__}
{len(self)} measurements of {dict_str}"""

    def __init__(self, system: System) -> None:
        super().__init__()

        self.system = system
        self.history = self.History(system)
        self.metrics = al.metrics.MetricList()

    def save(self, h5_handle) -> None:
        self.history.save(h5_handle.create_group("history"))
        self.metrics.save(h5_handle.create_group("metrics"))

    @abstractmethod
    def apply_measurement(self, tf_x, tf_y):
        ...

    def plot_measurement_curve(self, label):
        return self.system.measurement_curve(self, label)


class BayesLearner(Learner):
    prior = ...
    likelihood = ...
    posterior = ...
    # history = ...

    def __init__(self, system):
        super().__init__(system)

        if type(system).__name__ == al.systems.Qubits.__name__:
            self.likelihood = al.distributions.ConditionalBernoulli(
                dim=1, tf_p_func=system.tf_f
            )

        elif type(system).__name__ == al.systems.QubitsBinomial.__name__:
            self.likelihood = al.distributions.ConditionalBinomial(
                dim=1, tf_p_func=system.tf_f, n_counts=float(system.n_counts)
            )
        else:
            self.likelihood = al.distributions.ConditionalGaussianLikelihood(
                dim=self.system.dim_y,
                tf_mu_func=self.system.tf_f,
                sigma=self.system.p.sigma_noise,
            )

        self.metrics.append(al.metrics.Covariance(self))
        self.metrics.append(
            al.metrics.OutputKullbackLeibler(learner=self, x_range=self.system.x_range)
        )

        self.discreteLearner = None

    @classmethod
    def from_default(cls, system: al.systems.System, num_bijectors=4):
        learner = cls(system)
        learner.prior = learner._default_prior()
        learner.posterior = al.distributions.ConditionalNormalizingFlow(
            dim=learner.system.dim_lambda,
            dim_conditions=learner.system.dim_y,
            conditional_arguments_max=system.max_range,
            num_bijectors=num_bijectors,
        )

        # if al.config.cfg.LEARNER.DISCRETE.ENABLED:
        #     if not (system.dim_x != 1 or system.dim_lambda != 2 or system.dim_y > 2):
        #         learner.discreteLearner = al.learners.DiscreteLearner.from_default(learner,
        #         x_range=al.config.cfg.LEARNER.DISCRETE.X_RANGE,
        #         y1_range=al.config.cfg.LEARNER.DISCRETE.Y1_RANGE,
        #         y2_range=al.config.cfg.LEARNER.DISCRETE.Y2_RANGE,
        #         lambda1_range=al.config.cfg.LEARNER.DISCRETE.LAMBDA1_RANGE,
        #         lambda2_range=al.config.cfg.LEARNER.DISCRETE.LAMBDA2_RANGE)
        return learner

    @classmethod
    def from_h5_handle(cls, system: al.systems.System, h5_handle, load_discrete=False):
        learner = cls(system)
        learner.history.load(h5_handle["history"])
        if (
            h5_handle["prior"].attrs[al.utils.CLASS_NAME_PROPERTY]
            == al.distributions.ConditionedNormalizingFlow.__name__
        ):
            # Todo: can be transformed in @classmethod of distribution
            num_bijectors = h5_handle["prior"].attrs["num_bijectors"]
            learner.prior = al.distributions.ConditionalNormalizingFlow(
                dim=learner.system.dim_lambda,
                dim_conditions=learner.system.dim_y,
                conditional_arguments_max=system.max_range,
                num_bijectors=num_bijectors,
            ).condition_at(
                tf.convert_to_tensor(learner.history.y[-1], K.backend.floatx())
            )
            learner.prior.load(h5_handle["prior"])
        elif (
            h5_handle["prior"].attrs[al.utils.CLASS_NAME_PROPERTY]
            == al.distributions.ConditionedMultivariateGaussian.__name__
        ):
            n_neurons = h5_handle["posterior"].attrs["n_neurons"]
            n_layers = h5_handle["posterior"].attrs["n_layers"]
            learner.prior = al.distributions.ConditionalMultivariateGaussian(
                dim=learner.system.dim_lambda,
                dim_conditions=learner.system.dim_y,
                n_neurons=n_neurons,
                n_layers=n_layers,
            ).condition_at(
                tf.convert_to_tensor(learner.history.y[-1], K.backend.floatx())
            )
        else:
            learner.prior = learner._default_prior()

        if (
            h5_handle["posterior"].attrs[al.utils.CLASS_NAME_PROPERTY]
            == al.distributions.ConditionalNormalizingFlow.__name__
        ):
            num_bijectors = h5_handle["posterior"].attrs["num_bijectors"]
            learner.posterior = al.distributions.ConditionalNormalizingFlow(
                dim=learner.system.dim_lambda,
                dim_conditions=learner.system.dim_y,
                conditional_arguments_max=system.max_range,
                num_bijectors=num_bijectors,
            )
            learner.posterior.load(h5_handle["posterior"])
        elif (
            h5_handle["posterior"].attrs[al.utils.CLASS_NAME_PROPERTY]
            == al.distributions.ConditionalMultivariateGaussian.__name__
        ):
            n_neurons = h5_handle["posterior"].attrs["n_neurons"]
            n_layers = h5_handle["posterior"].attrs["n_layers"]
            learner.posterior = al.distributions.ConditionalMultivariateGaussian(
                dim=learner.system.dim_lambda,
                dim_conditions=learner.system.dim_y,
                n_neurons=n_neurons,
                n_layers=n_layers,
            )
            learner.posterior.load(h5_handle["posterior"])

        if load_discrete and "discreteLearner" in h5_handle.keys():
            learner.discreteLearner = al.learners.DiscreteLearner.from_h5_handle(
                learner=learner, h5_handle=h5_handle["discreteLearner"]
            )
        return learner

    def _default_prior(self):
        # TODO: add setting to set mu and sigma
        # The qubit example uses mu=1 as prior,
        # since the true frecuencies are detuned around 1.
        return al.distributions.IndependentGaussian(self.system.dim_lambda)

    def apply_measurement(self, tf_x, tf_y):
        conditioned_posterior = self.posterior.condition_at(tf_y)
        al.logger.info("Updating prior.")
        self.prior = conditioned_posterior
        self.metrics.recompile_tf_graphs()
        self.advisor.recompile_tf_graph()

        self.history.append(tf_x.numpy(), tf_y.numpy())
        if self.discreteLearner is not None:
            self.discreteLearner.apply_measurement(
                self.history.x[-1], self.history.y[-1]
            )

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        self.prior.save(h5_handle.create_group("prior"))
        self.posterior.save(h5_handle.create_group("posterior"))
        if self.discreteLearner is not None:
            self.discreteLearner.save(h5_handle.create_group("discreteLearner"))

    def __repr__(self) -> str:
        return f"""{self.__class__}
Prior:\t{self.prior.__repr__()}
Likelihood:\t{self.likelihood.__repr__()}
Posterior:\t{self.posterior.__repr__()}
History:\t{self.history.__repr__()}
Metrics:\t{self.metrics.__repr__()}
System:\t{self.system.__repr__()}
    """


class DiscreteLearner(Learner):
    """
    Numerical implementation of Bayesian parameter inference useful for testing.
    """

    def __init__(
        self, learner, x_range, y1_range, y2_range, lambda1_range, lambda2_range
    ):
        super().__init__(learner.system)

        self.learner = learner

        if (
            self.system.dim_x != 1
            or self.system.dim_lambda != 2
            or self.system.dim_y > 2
        ):
            raise NotImplementedError

        # Possible measurement position
        self.xs = np.linspace(*x_range[:-1], int(x_range[-1]))
        self.delta_xs = self.xs[1] - self.xs[0]

        # Possible measurement outcomes
        self.ys_1 = np.linspace(*y1_range[:-1], int(y1_range[-1]))
        self.delta_ys_1 = self.ys_1[1] - self.ys_1[0]

        if self.system.dim_y == 1:
            self.ys_2 = np.array([0])
            self.delta_ys_2 = 1
        elif self.system.dim_y == 2:
            self.ys_2 = np.linspace(*y2_range[:-1], int(y2_range[-1]))
            self.delta_ys_2 = self.ys_2[1] - self.ys_2[0]
        else:
            raise NotImplementedError

        self.y1, self.y2 = np.meshgrid(self.ys_1, self.ys_2)
        self.ys = np.dstack([self.y1, self.y2])

        self.lambdas_1 = np.linspace(*lambda1_range[:-1], int(lambda1_range[-1]))
        self.delta_lambdas_1 = self.lambdas_1[1] - self.lambdas_1[0]

        self.lambdas_2 = np.linspace(*lambda2_range[:-1], int(lambda2_range[-1]))
        self.delta_lambdas_2 = self.lambdas_2[1] - self.lambdas_2[0]
        self.l1, self.l2 = np.meshgrid(self.lambdas_1, self.lambdas_2)
        self.lambdas = np.dstack([self.l1, self.l2])

        self.p_y_given_lambda_x = self.get_p_y_given_lambda(
            self.ys[:, :, None, None, None, :],
            self.xs[None, None, :, None, None],
            self.lambdas[None, None],
        )

        self.p_lambdas_0 = self.learner.prior.prob(self.lambdas).numpy()

        self.learner.metrics.append(
            al.metrics.DiscreteExpectedInformationGainCurve(self, self.xs)
        )

        al.logger.info(
            "Initialized discretized version of Learner of numerical comparison with variational bound."
        )

    @classmethod
    def from_h5_handle(cls, learner, h5_handle):
        x_range = h5_handle.attrs["x_range"]
        y1_range = h5_handle.attrs["y1_range"]
        y2_range = h5_handle.attrs["y2_range"]
        lambda1_range = h5_handle.attrs["lambda1_range"]
        lambda2_range = h5_handle.attrs["lambda2_range"]

        discreteLearner = cls(
            learner, x_range, y1_range, y2_range, lambda1_range, lambda2_range
        )

        discreteLearner.history.load(h5_handle["history"])
        discreteLearner.p_lambdas = h5_handle["p_lambdas"][:]
        return discreteLearner

    @classmethod
    def from_default(cls, learner, *args, **kwargs):
        discreteLearner = cls(learner, *args, **kwargs)

        discreteLearner.p_lambdas = discreteLearner.p_lambdas_0.copy()
        return discreteLearner

    # Likelihood
    def get_p_y_given_lambda(self, y, x, lambda_):
        if self.system.dim_y == 1:
            return self.gaussian_1d(y - self.f(x, lambda_), self.system.p.sigma_noise)
        elif self.system.dim_y == 2:
            return self.gaussian_2d(y - self.f(x, lambda_), self.system.p.sigma_noise)

    def gaussian_2d(self, x, sigma):
        return (
            1
            / ((2 * np.pi) * sigma**2)
            * np.exp(-np.sum(x**2, -1) / (2 * sigma**2))
        )

    def gaussian_1d(self, x, sigma):
        return (
            1
            / np.sqrt((2 * np.pi) * sigma**2)
            * np.exp(-x[..., 0] ** 2 / (2 * sigma**2))
        )

    def np_diag(self, batched_mat):
        return batched_mat[..., None] * np.eye(batched_mat.shape[-1])

    def get_H_mat(self, lambda_):
        tf_lambda = tf.convert_to_tensor(lambda_, K.backend.floatx())
        return self.system.tf_get_H_mat(tf_lambda).numpy()

    def f(self, x_in, lambda_):
        tf_x = tf.convert_to_tensor(x_in, K.backend.floatx())
        tf_lambda = tf.convert_to_tensor(lambda_, K.backend.floatx())
        return self.system.tf_f(tf_x, tf_lambda).numpy()

    def IG(self):
        p_y_given_x = np.sum(
            self.p_y_given_lambda_x
            * self.p_lambdas[None, None, :]
            * self.delta_lambdas_1
            * self.delta_lambdas_2,
            (-1, -2),
        )
        # Probabilities should be normalized!
        test_prob_norm = p_y_given_x.sum((0, 1)) * self.delta_ys_1 * self.delta_ys_2
        assert np.allclose(
            test_prob_norm, 1, atol=1e-2, rtol=1e-2
        ), f"Probabilities should be normalized {test_prob_norm}"

        H_y_given_lambda = (
            -np.sum(
                (
                    sp.special.xlogy(
                        self.p_y_given_lambda_x * self.p_lambdas[None, None, :],
                        self.p_y_given_lambda_x,
                    )
                ),
                (0, 1, -1, -2),
            )
            * self.delta_lambdas_1
            * self.delta_lambdas_2
            * self.delta_ys_1
            * self.delta_ys_2
        )

        H_y = (
            -np.sum(sp.special.xlogy(p_y_given_x, p_y_given_x), (0, 1))
            * self.delta_ys_1
            * self.delta_ys_2
        )
        return H_y - H_y_given_lambda

    def apply_measurement(self, measure_at_x, measure_at_y):
        self.p_y_given_lambda = self.get_p_y_given_lambda(
            measure_at_y, measure_at_x, self.lambdas
        )
        self.p_y = (
            np.sum(self.p_y_given_lambda * self.p_lambdas, (-1, -2))
            * self.delta_lambdas_1
            * self.delta_lambdas_2
        )
        self.p_lambdas = self.p_y_given_lambda * self.p_lambdas / self.p_y

        self.history.append(measure_at_x, measure_at_y)

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["x_range"] = self.xs[0], self.xs[-1], len(self.xs)
        h5_handle.attrs["y1_range"] = self.ys_1[0], self.ys_1[-1], len(self.ys_1)
        h5_handle.attrs["y2_range"] = self.ys_2[0], self.ys_2[-1], len(self.ys_2)
        h5_handle.attrs["lambda1_range"] = (
            self.lambdas_1[0],
            self.lambdas_1[-1],
            len(self.lambdas_1),
        )
        h5_handle.attrs["lambda2_range"] = (
            self.lambdas_2[0],
            self.lambdas_2[-1],
            len(self.lambdas_2),
        )

        h5_handle.create_dataset("p_lambdas", data=self.p_lambdas)
        self.history.save(h5_handle.create_group("history"))
