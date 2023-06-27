import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import activelearning as al
from activelearning.normalizing_flows import MaskedAutoregressiveFlowNew, NormalizingFlows
import h5py

from tqdm import trange
import logging

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

from activelearning.config import cfg


class Distribution(ABC):
    """General helper for a distribution.
    
    Wraps some TensorFlow probability functions and adds checkpointing support."""
    @abstractmethod
    def sample(self, N_samples, **kwargs): ...

    @abstractmethod
    def prob(self, samples, **kwargs): ...

    @abstractmethod
    def log_prob(self, samples, **kwargs): ...

    def __call__(self, *args):
        return self.sample(*args)

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__


class IndependentGaussian(Distribution):

    def __init__(self, dim, mu=0., sigma=1.):
        self._dim = dim
        self._distribution = tfp.distributions.Sample(
            tfp.distributions.Normal(loc=mu,
                                     scale=sigma),
            sample_shape=(dim,))

    def sample(self, N_samples):
        return self._distribution.sample(N_samples)

    def prob(self, samples):
        return self._distribution.prob(samples)

    def log_prob(self, samples):
        return self._distribution.log_prob(samples)

    def __repr__(self) -> str:
        return f"{self.__class__}"


class MultivariateGaussian(Distribution):

    def __init__(self, dim, mu, sigma):
        self._dim = dim
        self._distribution = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                                      scale_tril=tf.linalg.cholesky(sigma)
                                                                      )

    def sample(self, N_samples):
        return self._distribution.sample(N_samples)

    def prob(self, samples):
        return self._distribution.prob(samples)

    def log_prob(self, samples):
        return self._distribution.log_prob(samples)

    def __repr__(self) -> str:
        return f"{self.__class__}"


class ConditionalGaussian(Distribution):

    def __init__(self, dim):
        self._dim = dim
        self._distribution = K.Sequential([
            K.layers.InputLayer(input_shape=(tfp.layers.IndependentNormal.params_size(self._dim),)),
            tfp.layers.IndependentNormal(self._dim)
        ])

    def sample(self, N_samples, mu, sigma):
        _softplus_sigma = tf.math.log(tf.exp(sigma)-1)
        return self._distribution(tf.concat([mu, _softplus_sigma], -1)).sample(N_samples)

    def prob(self, samples, mu, sigma):
        _softplus_sigma = tf.math.log(tf.exp(sigma)-1)

        return self._distribution(tf.concat([mu, _softplus_sigma], -1)).prob(samples)

    def log_prob(self, samples, mu, sigma):
        _softplus_sigma = tf.math.log(tf.exp(sigma)-1)
        return self._distribution(tf.concat([mu, _softplus_sigma], -1)).log_prob(samples)

    def __repr__(self) -> str:
        return f"""{self.__class__}"""


class ConditionalBernoulli(Distribution):

    def __init__(self, dim, tf_p_func) -> None:
        self._dim = dim
        self.tf_p_func = tf_p_func
        self._distribution = K.Sequential([
            K.layers.InputLayer(input_shape=(self._dim,)),
            tfp.layers.IndependentBernoulli(self._dim)
        ])

    def sample(self, N_samples,  **kwargs):
        # p = probability of a 1 event
        p = self.tf_p_func(**kwargs)[..., 1]
        logit = tf.math.log(p) - tf.math.log(1-p)
        return self._distribution(logit).sample(N_samples)

    def prob(self, samples,  **kwargs):
        p = self.tf_p_func(**kwargs)[..., 1]
        logit = tf.math.log(p) - tf.math.log(1-p)
        return self._distribution(logit).prob(samples)

    def log_prob(self, samples,  **kwargs):
        p = self.tf_p_func(**kwargs)[..., 1]
        logit = tf.math.log(p) - tf.math.log(1-p)
        return self._distribution(logit).log_prob(samples)


class ConditionalBinomial(Distribution):
    
    def __init__(self, dim, tf_p_func, n_counts) -> None:
        self._dim = dim
        self.tf_p_func = tf_p_func
        self.n_counts = n_counts
        
        self._distribution = K.Sequential([
            K.layers.InputLayer(input_shape=(self._dim,)),
            tfp.layers.DistributionLambda(
                make_distribution_fn= lambda p: tfp.distributions.Binomial(n_counts, probs=p)
            )
        ])

    def sample(self, N_samples,  **kwargs):
        p = self.tf_p_func(**kwargs)[..., 0] # take qubit #0
        return self._distribution(p).sample(N_samples)/self.n_counts

    def prob(self, samples,  **kwargs):
        p = self.tf_p_func(**kwargs)[..., 0]
        return self._distribution(p).prob(samples*self.n_counts)

    def log_prob(self, samples,  **kwargs):
        p = self.tf_p_func(**kwargs)[..., 0]
        return self._distribution(p).log_prob(samples*self.n_counts)
    
    
class ConditionalGaussianLikelihood(ConditionalGaussian):

    def __init__(self, dim, tf_mu_func, sigma):
        # tf_mu_func: mu = f(x,lambda) for active learning
        self._dim = dim
        self._mu_func = tf_mu_func
        self._sigma = tf.convert_to_tensor(sigma, K.backend.floatx())

        super().__init__(self._dim)

    def sample(self, N_samples, **kwargs):
        mu = self._mu_func(**kwargs)
        sigma = tf.fill(mu.shape, self._sigma)
        return super().sample(N_samples, mu, sigma)

    def prob(self, samples, **kwargs):
        mu = self._mu_func(**kwargs)
        sigma = tf.fill(mu.shape, self._sigma)
        return super().prob(samples, mu, sigma)

    def log_prob(self, samples, **kwargs):
        mu = self._mu_func(**kwargs)
        sigma = tf.fill(mu.shape, self._sigma)
        return super().log_prob(samples, mu, sigma)

    def __repr__(self) -> str:
        return f"{self.__class__} with sigma = {self._sigma.numpy():2.3f}"


class ConditionalMultivariateGaussian(Distribution):
    def __init__(self, dim, dim_conditions, n_layers=2, n_neurons=30):
        self._dim = dim
        self._dim_conditions = dim_conditions
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # ToDo: more layers not implemented yet
        self._distribution = K.Sequential([
            K.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self._dim),
                           input_shape=(self._dim_conditions,)),
            tfp.layers.MultivariateNormalTriL(self._dim)
        ])

    def sample(self, N_samples, conditional_arguments):
        return self._distribution(conditional_arguments[None]).sample(N_samples)[..., 0, :]

    def prob(self, samples, conditional_arguments):
        return self._distribution(conditional_arguments[None]).prob(samples)[..., 0, :]

    def log_prob(self, samples, conditional_arguments):
        if len(conditional_arguments.shape) == 2:  # problems with the shape; implement also in other functions!
           # shape can be (dimY) or (n_samples, dimY)
            return self._distribution(conditional_arguments).log_prob(samples)
        else:
            return self._distribution(conditional_arguments[None]).log_prob(samples)  # [...,0,:]

    def condition_at(self, tf_fixed_condition):
        return al.distributions.ConditionedMultivariateGaussian(
            conditional_multivariate_gaussian=self,
            tf_fixed_conditions=tf_fixed_condition
        )

    @property
    def weights(self):
        return self._distribution.get_weights()

    @weights.setter
    def weights(self, weights):
        return self._distribution.set_weights(weights)

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs["n_neurons"] = self.n_neurons
        h5_handle.attrs["n_layers"] = self.n_layers
        al.utils.save_nn_weights(net_weights=self.weights,
                                 h5_handle=h5_handle)

    def load(self, h5_handle):
        self.weights = al.utils.load_nn_weights(h5_handle)

    @property
    def trainable_variables(self):
        return self._distribution.trainable_variables

    def __repr__(self) -> str:
        return f"{self.__class__}"


class ConditionedMultivariateGaussian(ConditionalMultivariateGaussian):
    def __init__(self, conditional_multivariate_gaussian, tf_fixed_conditions):
        super().__init__(dim=conditional_multivariate_gaussian._dim,
                         dim_conditions=conditional_multivariate_gaussian._dim_conditions,
                         n_neurons=conditional_multivariate_gaussian.n_neurons,
                         n_layers=conditional_multivariate_gaussian.n_layers)

        self.weights = conditional_multivariate_gaussian.weights

        self.tf_fixed_conditions = tf_fixed_conditions[None]
        assert tf_fixed_conditions.shape == conditional_multivariate_gaussian._dim_conditions, \
            f"Shape of condition is wrong: expected {conditional_multivariate_gaussian._dim_conditions}, got {tf_fixed_conditions.shape[0]}."

    def sample(self, N_samples):
        return self._distribution(self.tf_fixed_conditions).sample(N_samples)[..., 0, :]

    def prob(self, samples):
        return self._distribution(self.tf_fixed_conditions).prob(samples)

    def log_prob(self, samples):
        return self._distribution(self.tf_fixed_conditions).log_prob(samples)

    def __repr__(self) -> str:
        return f"{self.__class__} with tf_fixed_condition at {self.tf_fixed_conditions.numpy()}"


class ConditionalNormalizingFlow(Distribution):

    def nf_cond_args(self, args):
        return {'arf': {'conditional_input': args}}

    def __init__(self, dim, dim_conditions,  conditional_arguments_max, num_bijectors=4):
        self._dim = dim
        self._dim_conditions = dim_conditions
        self.num_bijectors = num_bijectors
        self.conditional_arguments_max = conditional_arguments_max

        self._base_distribution = tfp.distributions.Sample(
            tfp.distributions.Normal(loc=0, scale=1),
            sample_shape=[self._dim])

        flow_bijector = NormalizingFlows.get_deep_normalizing_flow(
            n_inputs=self._dim,
            num_bijectors=self.num_bijectors,
            dim_y=self._dim_conditions)

        self._distribution = flow_bijector(self._base_distribution)

        # initialize the weights of the network
        # by performing a useless forward pass
        self._distribution.prob(
            np.array([np.arange(self._dim)]),
            bijector_kwargs=self.nf_cond_args(tf.zeros([1, self._dim_conditions], dtype=K.backend.floatx()))
        )

    def sample(self, N_samples, conditional_arguments):
        conditional_arguments /= self.conditional_arguments_max
        return self._distribution.sample(N_samples,
                                         bijector_kwargs=self.nf_cond_args(conditional_arguments))

    def prob(self, samples, conditional_arguments):
        conditional_arguments /= self.conditional_arguments_max
        return self._distribution.prob(samples,
                                       bijector_kwargs=self.nf_cond_args(conditional_arguments))

    def log_prob(self, samples, conditional_arguments):
        conditional_arguments /= self.conditional_arguments_max
        return self._distribution.log_prob(samples,
                                           bijector_kwargs=self.nf_cond_args(conditional_arguments))

    def condition_at(self, tf_fixed_condition):
        tf_fixed_condition /= self.conditional_arguments_max
        return al.distributions.ConditionedNormalizingFlow(
            conditional_normalizing_flow=self,
            tf_fixed_conditions=tf_fixed_condition
        )

    def save(self, h5_handle):
        super().save(h5_handle)
        h5_handle.attrs["num_bijectors"] = self.num_bijectors
        NormalizingFlows.save_weights(net_weights=self.weights,
                                      h5_handle=h5_handle)

    def load(self, h5_handle):
        self.weights = NormalizingFlows.load_weights(h5_handle)

    @property
    def weights(self):
        return NormalizingFlows.get_weights(self._distribution.bijector)

    @weights.setter
    def weights(self, weights):
        return NormalizingFlows.set_weights(self._distribution.bijector, weights)

    @property
    def trainable_variables(self):
        return self._distribution.trainable_variables

    def __repr__(self) -> str:
        return f"{self.__class__}"


class ConditionedNormalizingFlow(ConditionalNormalizingFlow):
    def __init__(self, conditional_normalizing_flow, tf_fixed_conditions):
        super().__init__(dim=conditional_normalizing_flow._dim,
                         dim_conditions=conditional_normalizing_flow._dim_conditions,
                         conditional_arguments_max=1,
                         num_bijectors=conditional_normalizing_flow.num_bijectors)

        self.weights = conditional_normalizing_flow.weights

        self.tf_fixed_conditions = tf_fixed_conditions/self.conditional_arguments_max
        assert tf_fixed_conditions.shape == conditional_normalizing_flow._dim_conditions, \
            f"Shape of condition is wrong: expected {conditional_normalizing_flow._dim_conditions}, got {tf_fixed_conditions.shape[0]}."

    def sample(self, N_samples):
        return self._distribution.sample(N_samples,
                                         bijector_kwargs=self.nf_cond_args(self.tf_fixed_conditions))

    def prob(self, samples):
        return self._distribution.prob(samples,
                                       bijector_kwargs=self.nf_cond_args(self.tf_fixed_conditions))

    def log_prob(self, samples):
        return self._distribution.log_prob(samples,
                                           bijector_kwargs=self.nf_cond_args(self.tf_fixed_conditions))

    def __repr__(self) -> str:
        return f"{self.__class__} with tf_fixed_condition at {self.tf_fixed_conditions.numpy()}"
