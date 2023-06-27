# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import scale as scale_lib
from tensorflow_probability.python.bijectors import shift as shift_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient

from tensorflow.python.util import deprecation

import numpy as np
import tensorflow_probability as tfp
import tensorflow.keras as K
import h5py


class MaskedAutoregressiveFlowNew(bijector_lib.Bijector):
    """Affine MaskedAutoregressiveFlow bijector.

    The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
    relatively simple framework for user-specified (deep) architectures to learn a
    distribution over continuous events.  Regarding terminology,

      'Autoregressive models decompose the joint density as a product of
      conditionals, and model each conditional in turn.  Normalizing flows
      transform a base density (e.g. a standard Gaussian) into the target density
      by an invertible transformation with tractable Jacobian.'
      [(Papamakarios et al., 2016)][3]

    In other words, the 'autoregressive property' is equivalent to the
    decomposition, `p(x) = prod{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d }`
    where `perm` is some permutation of `{0, ..., d}`.  In the simple case where
    the permutation is identity this reduces to:
    `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`.

    In TensorFlow Probability, 'normalizing flows' are implemented as
    `tfp.bijectors.Bijector`s.  The `forward` 'autoregression' is implemented
    using a `tf.while_loop` and a deep neural network (DNN) with masked weights
    such that the autoregressive property is automatically met in the `inverse`.

    A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
    (expensive) forward-mode calculation to draw samples and the (cheap)
    reverse-mode calculation to compute log-probabilities.  Conversely, a
    `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
    the (expensive) forward-mode calculation to compute log-probabilities and the
    (cheap) reverse-mode calculation to compute samples.  See 'Example Use'
    [below] for more details.

    Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
    (a sequence of) affine transformations.  A 'valid' `shift_and_log_scale_fn`
    must compute each `shift` (aka `loc` or 'mu' in [Germain et al. (2015)][1])
    and `log(scale)` (aka 'alpha' in [Germain et al. (2015)][1]) such that each
    are broadcastable with the arguments to `forward` and `inverse`, i.e., such
    that the calculations in `forward`, `inverse` [below] are possible.

    For convenience, `tfp.bijectors.AutoregressiveNetwork` is offered as a
    possible `shift_and_log_scale_fn` function.  It implements the MADE
    architecture [(Germain et al., 2015)][1].  MADE is a feed-forward network that
    computes a `shift` and `log(scale)` using masked dense layers in a deep
    neural network. Weights are masked to ensure the autoregressive property. It
    is possible that this architecture is suboptimal for your task. To build
    alternative networks, either change the arguments to
    `tfp.bijectors.AutoregressiveNetwork` or use some other architecture, e.g.,
    using `tf.keras.layers`.

    Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
    enforces the 'autoregressive property'.

    Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
    semantics, the forward transformation is

    ```python
    def forward(x):
      y = zeros_like(x)
      event_size = x.shape[-event_dims:].num_elements()
      for _ in range(event_size):
        shift, log_scale = shift_and_log_scale_fn(y)
        y = x * tf.exp(log_scale) + shift
      return y
    ```

    and the inverse transformation is

    ```python
    def inverse(y):
      shift, log_scale = shift_and_log_scale_fn(y)
      return (y - shift) / tf.exp(log_scale)
    ```

    Notice that the `inverse` does not need a for-loop.  This is because in the
    forward pass each calculation of `shift` and `log_scale` is based on the `y`
    calculated so far (not `x`).  In the `inverse`, the `y` is fully known, thus
    is equivalent to the scaling used in `forward` after `event_size` passes,
    i.e., the 'last' `y` used to compute `shift`, `log_scale`.  (Roughly speaking,
    this also proves the transform is bijective.)

    The `bijector_fn` argument allows specifying a more general coupling relation,
    such as the LSTM-inspired activation from [4], or Neural Spline Flow [5].  It
    must logically operate on each element of the input individually, and still
    obey the 'autoregressive property' described above.  The forward
    transformation is

    ```python
    def forward(x):
      y = zeros_like(x)
      event_size = x.shape[-event_dims:].num_elements()
      for _ in range(event_size):
        bijector = bijector_fn(y)
        y = bijector.forward(x)
      return y
    ```

    and inverse transformation is

    ```python
    def inverse(y):
        bijector = bijector_fn(y)
        return bijector.inverse(y)
    ```

    #### Examples

    ```python
    tfd = tfp.distributions
    tfb = tfp.bijectors

    dims = 2

    # A common choice for a normalizing flow is to use a Gaussian for the base
    # distribution.  (However, any continuous distribution would work.) Here, we
    # use `tfd.Sample` to create a joint Gaussian distribution with diagonal
    # covariance for the base distribution (note that in the Gaussian case,
    # `tfd.MultivariateNormalDiag` could also be used.)
    maf = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
        bijector=tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2, hidden_units=[512, 512])))

    x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
    maf.log_prob(x)   # Almost free; uses Bijector caching.
    # Cheap; no `tf.while_loop` despite no Bijector caching.
    maf.log_prob(tf.zeros(dims))

    # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
    # Flow [(Kingma et al., 2016)][2]:
    iaf = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
        bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2, hidden_units=[512, 512]))))

    x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
    iaf.log_prob(x)   # Almost free; uses Bijector caching.
    # Expensive; uses `tf.while_loop`, no Bijector caching.
    iaf.log_prob(tf.zeros(dims))

    # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
    # poor choice.  Here's an example of using a 'shift only' version and with a
    # different number/depth of hidden layers.
    made = tfb.AutoregressiveNetwork(params=1, hidden_units=[32])
    maf_no_scale_hidden2 = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
        bijector=tfb.MaskedAutoregressiveFlow(
            lambda y: (made(y)[..., 0], None),
            is_constant_jacobian=True))
    maf_no_scale_hidden2._made = made  # Ensure maf_no_scale_hidden2.trainable
    # NOTE: The last line ensures that maf_no_scale_hidden2.trainable_variables
    # will include all variables from `made`.
    ```

    #### Variable Tracking

    NOTE: Like all subclasses of `tfb.Bijector`, `tfb.MaskedAutoregressiveFlow`
    subclasses `tf.Module` for variable tracking.

    A `tfb.MaskedAutoregressiveFlow` instance saves a reference to the values
    passed as `shift_and_log_scale_fn` and `bijector_fn` to its constructor.
    Thus, for most values passed as `shift_and_log_scale_fn` or `bijector_fn`,
    variables referenced by those values will be found and tracked by the
    `tfb.MaskedAutoregressiveFlow` instance.  Please see the `tf.Module`
    documentation for further details.

    However, if the value passed to `shift_and_log_scale_fn` or `bijector_fn` is a
    Python function, then `tfb.MaskedAutoregressiveFlow` cannot automatically
    track variables used inside `shift_and_log_scale_fn` or `bijector_fn`.  To get
    `tfb.MaskedAutoregressiveFlow` to track such variables, either:

     1. Replace the Python function with a `tf.Module`, `tf.keras.Layer`,
        or other callable object through which `tf.Module` can find variables.

     2. Or, add a reference to the variables to the `tfb.MaskedAutoregressiveFlow`
        instance by setting an attribute -- for example:
        ````
        made1 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
        made2 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
        maf = tfb.MaskedAutoregressiveFlow(lambda y: (made1(y), made2(y) + 1.))
        maf._made_variables = made1.variables + made2.variables
        ````

    #### References

    [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
         Masked Autoencoder for Distribution Estimation. In _International
         Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

    [2]: Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya
         Sutskever, and Max Welling. Improving Variational Inference with Inverse
         Autoregressive Flow. In _Neural Information Processing Systems_, 2016.
         https://arxiv.org/abs/1606.04934

    [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
         Autoregressive Flow for Density Estimation. In _Neural Information
         Processing Systems_, 2017. https://arxiv.org/abs/1705.07057

    [4]: Diederik P Kingma, Tim Salimans, Max Welling. Improving Variational
         Inference with Inverse Autoregressive Flow. In _Neural Information
         Processing Systems_, 2016. https://arxiv.org/abs/1606.04934

    [5]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
         Spline Flows, 2019. http://arxiv.org/abs/1906.04032
    """

    def __init__(
        self,
        shift_and_log_scale_fn=None,
        bijector_fn=None,
        is_constant_jacobian=False,
        validate_args=False,
        unroll_loop=False,
        event_ndims=1,
        name=None,
    ):
        """Creates the MaskedAutoregressiveFlow bijector.

        Args:
          shift_and_log_scale_fn: Python `callable` which computes `shift` and
            `log_scale` from the inverse domain (`y`). Calculation must respect the
            'autoregressive property' (see class docstring). Suggested default
            `tfb.AutoregressiveNetwork(params=2, hidden_layers=...)`.
            Typically the function contains `tf.Variables`. Returning `None` for
            either (both) `shift`, `log_scale` is equivalent to (but more efficient
            than) returning zero. If `shift_and_log_scale_fn` returns a single
            `Tensor`, the returned value will be unstacked to get the `shift` and
            `log_scale`: `tf.unstack(shift_and_log_scale_fn(y), num=2, axis=-1)`.
          bijector_fn: Python `callable` which returns a `tfb.Bijector` which
            transforms event tensor with the signature
            `(input, **condition_kwargs) -> bijector`. The bijector must operate on
            scalar events and must not alter the rank of its input. The
            `bijector_fn` will be called with `Tensors` from the inverse domain
            (`y`). Calculation must respect the 'autoregressive property' (see
            class docstring).
          is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
            implementation assumes `log_scale` does not depend on the forward domain
            (`x`) or inverse domain (`y`) values. (No validation is made;
            `is_constant_jacobian=False` is always safe but possibly computationally
            inefficient.)
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
          unroll_loop: Python `bool` indicating whether the `tf.while_loop` in
            `_forward` should be replaced with a static for loop. Requires that
            the final dimension of `x` be known at graph construction time. Defaults
            to `False`.
          event_ndims: Python `integer`, the intrinsic dimensionality of this
            bijector. 1 corresponds to a simple vector autoregressive bijector as
            implemented by the `tfp.bijectors.AutoregressiveNetwork`, 2 might be
            useful for a 2D convolutional `shift_and_log_scale_fn` and so on.
          name: Python `str`, name given to ops managed by this object.

        Raises:
          ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
              are specified.
        """
        parameters = dict(locals())
        name = name or "masked_autoregressive_flow"
        with tf.name_scope(name) as name:
            self._unroll_loop = unroll_loop
            self._event_ndims = event_ndims
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError(
                    "Exactly one of `shift_and_log_scale_fn` and "
                    "`bijector_fn` should be specified."
                )
            if shift_and_log_scale_fn:

                def _bijector_fn(x, **condition_kwargs):
                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    if tf.is_tensor(params):
                        shift, log_scale = tf.unstack(params, num=2, axis=-1)
                    else:
                        shift, log_scale = params

                    bijectors = []
                    if shift is not None:
                        bijectors.append(shift_lib.Shift(shift))
                    if log_scale is not None:
                        # NOTE: change from the original tensorflow probability implementation
                        # The tanh activation will help preventing the blow up of the log_scale
                        bijectors.append(
                            scale_lib.Scale(log_scale=5.0 * tf.tanh(log_scale / 5.0))
                        )
                    return chain.Chain(bijectors, validate_event_size=False)

                bijector_fn = _bijector_fn

            if validate_args:
                bijector_fn = tfp.bijectors.masked_autoregressive._validate_bijector_fn(
                    bijector_fn
                )
            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn
            super().__init__(
                forward_min_event_ndims=self._event_ndims,
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _parameter_properties(cls, dtype):
        return dict()

    def _forward(self, x, **kwargs):
        static_event_size = tensorshape_util.num_elements(
            tensorshape_util.with_rank_at_least(x.shape, self._event_ndims)[
                -self._event_ndims :
            ]
        )

        if self._unroll_loop:
            if not static_event_size:
                raise ValueError(
                    "The final {} dimensions of `x` must be known at graph "
                    "construction time if `unroll_loop=True`. `x.shape: {!r}`".format(
                        self._event_ndims, x.shape
                    )
                )
            y = tf.zeros_like(x, name="y0")

            for _ in range(static_event_size):
                y = self._bijector_fn(y, **kwargs).forward(x)
            return y

        event_size = ps.reduce_prod(ps.shape(x)[-self._event_ndims :])
        y0 = tf.zeros_like(x, name="y0")
        # call the template once to ensure creation
        if not tf.executing_eagerly():
            _ = self._bijector_fn(y0, **kwargs).forward(y0)

        def _loop_body(y0):
            """While-loop body for autoregression calculation."""
            # Set caching device to avoid re-getting the tf.Variable for every while
            # loop iteration.
            with tf1.variable_scope(tf1.get_variable_scope()) as vs:
                if vs.caching_device is None and not tf.executing_eagerly():
                    vs.set_caching_device(lambda op: op.device)
                bijector = self._bijector_fn(y0, **kwargs)
            y = bijector.forward(x)
            return (y,)

        (y,) = tf.while_loop(
            cond=lambda _: True,
            body=_loop_body,
            loop_vars=(y0,),
            maximum_iterations=event_size,
        )
        return y

    def _inverse(self, y, **kwargs):
        bijector = self._bijector_fn(y, **kwargs)
        return bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y, **kwargs):
        return self._bijector_fn(y, **kwargs).inverse_log_det_jacobian(
            y, event_ndims=self._event_ndims
        )


# ============================================================================
#  Code for Deep Bayesian Experimental Design for Quantum Many-body Systems
# ============================================================================


class NormalizingFlows:
    ## Supports bijector indentation in a chain!
    def make_bijector_kwargs(bijector, name_to_kwargs):
        import re

        if hasattr(bijector, "bijectors"):
            return {
                b.name: NormalizingFlows.make_bijector_kwargs(b, name_to_kwargs)
                for b in bijector.bijectors
            }
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}

    def make_masked_autoregressive_flow(
        n_inputs, hidden_units=[64, 64], activation="relu", name="arf", dim_y=1
    ):
        made = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            event_shape=[n_inputs],
            hidden_units=hidden_units,
            activation=activation,
            conditional=True,
            conditional_event_shape=(dim_y,),
            kernel_initializer=K.initializers.VarianceScaling(0.1),
        )
        # The implementation of the MaskedAutoregressiveFlow class is derived
        # from the TensorFlow probability bijector
        return MaskedAutoregressiveFlowNew(shift_and_log_scale_fn=made, name=name)

    def get_deep_normalizing_flow(
        n_inputs,
        num_bijectors=6,
        hidden_units=[64, 64],
        activation="relu",
        name="arf",
        dim_y=1,
    ):
        bijectors = []

        for i in range(num_bijectors):
            masked_auto_i = NormalizingFlows.make_masked_autoregressive_flow(
                n_inputs,
                hidden_units=hidden_units,
                activation=activation,
                name=name,
                dim_y=dim_y,
            )
            bijectors.append(masked_auto_i)

            bijectors.append(
                (tfp.bijectors.Permute(permutation=np.roll(np.arange(n_inputs), 1, 0)))
            )

        flow_bijector = tfp.bijectors.Chain(list(reversed(bijectors[:-1])))
        return flow_bijector

    def get_weights(flow_bijector):
        net_weights = []
        for i in range(len(flow_bijector.bijectors)):
            if flow_bijector.bijectors[i].name == "arf":
                net_weights.append(
                    flow_bijector.bijectors[i]._shift_and_log_scale_fn.get_weights()
                )
        return net_weights

    def set_weights(flow_bijector, weights):
        current_index = 0
        for i in range(len(flow_bijector.bijectors)):
            if flow_bijector.bijectors[i].name == "arf":
                flow_bijector.bijectors[i]._shift_and_log_scale_fn.set_weights(
                    weights[current_index]
                )
                current_index += 1
        assert current_index == len(
            weights
        ), "You are using a network with different number of normalizing flow layers!"

    def save_weights(net_weights, h5_handle):
        for j in range(len(net_weights)):
            grpgrp = h5_handle.create_group(str(j))
            for i, lst in enumerate(net_weights[j]):
                grpgrp.create_dataset(str(i), data=lst)

    def load_weights(h5_handle):
        new_weights = []
        for keykey in h5_handle.keys():
            c_weights = []
            for key in h5_handle[keykey].keys():
                c_weights.append(h5_handle[keykey][key][:])
            new_weights.append(c_weights)
        return new_weights
