from argparse import ArgumentError
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import matplotlib.pyplot as plt
import scipy as sp
from typing import List
import scipy.special
from activelearning.normalizing_flows import NormalizingFlows
import activelearning as al
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

tf_complex = tf.complex128


class System(ABC):
    """
    Abstract System class.

    It describes a physical system by implementing a response function and a measure function (which also adds noise to the output).
    """
    @abstractmethod
    def tf_f(self, tf_x, tf_lambda): ...

    @abstractmethod
    def measure(self, x): ...

    @abstractproperty
    def tf_real_lambda(self): ...

    @abstractproperty
    def dim_x(self): ...

    @abstractproperty
    def dim_y(self): ...

    @property
    def dim_lambda(self):
        return len(self.tf_real_lambda)

    @property
    def max_range(self):
        # This would be y max range
        # It can be useful to normalize the conditional input
        # of the normalizing flow
        return 1


class Cavities(System):

    @dataclass
    class Parameters:
        j_coupling: List[float]
        k_dissipation_int: float
        k_dissipation_ext: float
        sigma_noise: float

        def __repr__(self) -> str:
            repr_str = ""
            for key, val in self.__dict__.items():
                repr_str += f"{key:25}:\t{val}\n"
            return repr_str

        def save(self, h5_handle):
            return al.utils.h5_save_dataclass(self, h5_handle.attrs)

    dim_x = 1
    dim_y = 2
    tf_real_lambda = np.array([])

    def __init__(self, real_lambda=[],
                 k_dissipation_int=0.5,
                 k_dissipation_ext=0.5,
                 j_coupling=None,  # 0=random couplings
                 sigma_noise=0.3,
                 dim_lambda=None, type_lambda=None,
                 x_range=None, **kwargs):
        if dim_lambda is None:
            dim_lambda = len(real_lambda)
        self.dim_system = dim_lambda

        if j_coupling is None:
            al.logger.error("Coupling not specified!")
            raise Exception
        elif not isinstance(j_coupling, np.ndarray) and not isinstance(j_coupling, list):
            if j_coupling == 0:
                # j_coupling = 2*np.random.uniform(0.5, 1.5, size=self.dim_system - 1)
                # TODO: This should become fixed for reproducibility in the paper!
                al.logger.warning("Random coupling assignment has been disabled for reproducibility over multiple runs.")
                j_coupling = np.array([2.7335782, 2.6151755, 1.9560152, 1.5678725, 2.6201243])

                if dim_lambda-1 != len(j_coupling):
                    raise Exception("Problem with the enforced coupling. Implemented only for a specific example. Please disable COUPLING_RANDOM")
            else:
                j_coupling = j_coupling*np.ones(shape=[self.dim_system - 1])

        self.p = self.Parameters(
            j_coupling=j_coupling,
            k_dissipation_int=k_dissipation_int,
            k_dissipation_ext=k_dissipation_ext,
            sigma_noise=sigma_noise
        )

        if real_lambda != []:
            real_lambda = np.array(real_lambda)
            if dim_lambda is not None and type_lambda is not None:
                al.logger.warning("real_lambda specified. Ignoring dim_lambda and type_lambda parameters")
        else:

            if dim_lambda is not None and type_lambda is not None:
                if type_lambda == "all_zero":
                    real_lambda = np.zeros(dim_lambda)
                elif type_lambda == "gaussian_random":
                    real_lambda = np.random.randn(dim_lambda)
                elif type_lambda == "barrier":
                    real_lambda = -np.ones(dim_lambda)*2
                    real_lambda[dim_lambda//2:] = 2
                else:
                    raise Exception("What do you want?")

        self.tf_real_lambda = tf.convert_to_tensor(real_lambda, K.backend.floatx())
        self.tf_coupling = tf.convert_to_tensor(self.p.j_coupling, K.backend.floatx())
        self.x_range = x_range  # x range list, e.g. for plots [x_min, x_max]

    @classmethod
    def from_h5_handle(cls, h5_handle):
        real_lambda = h5_handle.attrs["real_lambda"]
        x_range = list(h5_handle.attrs["x_range"])  # needed for plots
        return cls(**dict(h5_handle["p"].attrs), real_lambda=real_lambda, x_range=x_range)

    def measurement_curve(self, learner, label):
        return measurement_curve_2d(learner, label)

    def tf_diag(self, batched_mat):
        return batched_mat[..., None]*tf.eye(batched_mat.shape[-1])

    def tf_get_H_mat(self, tf_parameters):
        tf_H_diag = self.tf_diag(tf_parameters)
        # Choose j random to destroy symmetry in the transmission
        tf_j = self.tf_coupling*tf.ones([*tf_parameters.shape[:-1], self.dim_system - 1])
        tf_H_l = tf.roll(self.tf_diag(tf.concat([tf_j] + [tf.zeros([*tf_j.shape[:-1], 1])], -1)),
                         shift=1, axis=-2)
        tf_H_u = tf.roll(self.tf_diag(tf.concat([tf.zeros([*tf_j.shape[:-1], 1])] + [tf_j], -1)),
                         shift=-1, axis=-2)
        return tf_H_diag + tf_H_l + tf_H_u

    def tf_response_matrix(self, tf_x, tf_lambda):
        x = tf_x[..., None]*tf.ones([*tf_x.shape, self.dim_system])
        tf_H_mat = self.tf_get_H_mat(tf_lambda)

        k_dissipation_int_vect = np.array((self.dim_system)*[self.p.k_dissipation_int])
        k_dissipation_ext_vect = np.array(([self.p.k_dissipation_ext] + (self.dim_system-2)*[0]+[self.p.k_dissipation_ext]))
        k_dissipation_vect = k_dissipation_int_vect + k_dissipation_ext_vect
        tf_k_dissipation_mat = tf.linalg.diag(tf.convert_to_tensor(k_dissipation_vect, K.backend.floatx()))
        tf_k_dissipation_ext_mat = tf.linalg.diag(tf.convert_to_tensor(k_dissipation_ext_vect, "complex64"))

        return tf.eye(self.dim_system, dtype="complex64") - tf.matmul(
            tf_k_dissipation_ext_mat,
            tf.linalg.inv(tf.complex(tf_k_dissipation_mat/2, -(self.tf_diag(x) - tf_H_mat))))

    def tf_f(self, tf_x, tf_lambda):
        # tf_x has shape (n_samples,)
        # tf_lambda has shape (..., dim_lambda)

        tf_response_element = self.tf_response_matrix(tf_x, tf_lambda)[..., 0, -1]

        # out has shape (n_samples, dim_y)
        return tf.stack([tf.math.real(tf_response_element),
                         tf.math.imag(tf_response_element)], -1)

    def tf_fisher_function_plain(self, tf_x, tf_lambda):
        with tf.GradientTape() as tape:
            tape.watch(tf_lambda)
            tf_f = self.tf_f(tf_x, tf_lambda)
        grad_f = tape.jacobian(tf_f, tf_lambda)
        g1 = tf.reduce_sum(grad_f[..., None, :]*grad_f[..., :, None], axis=-3)
        # g1 shape: ... y_dim, lambda_dim, lambda_dim
        # sum over y_dim for the scalar product
        return g1/self.p.sigma_noise**2

    def measure(self, x):
        f = self.tf_f(x, self.tf_real_lambda).numpy()
        noise = np.random.normal(size=f.shape)*self.p.sigma_noise
        return f + noise

    @property
    def H_mat(self):
        return self.tf_get_H_mat(self.tf_real_lambda).numpy()

    def plot_response(self):
        PLOT_EXTENT = al.config.cfg.SYSTEM.X_RANGE
        xs = np.linspace(*PLOT_EXTENT, 150)
        fs = self.tf_f(xs, self.tf_real_lambda).numpy()

        xs_meas = np.repeat(xs, 5)
        ys_meas = self.measure(xs_meas)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Measurement function $Re f(x,\lambda^*)$")
        plt.plot(xs, fs[:, 0], linewidth=5)
        plt.scatter(xs_meas, ys_meas[:, 0], alpha=0.5, c="lightblue", s=10)
        plt.xlabel("$x$")
        plt.ylabel("$Real f(x,\lambda^*)$")

        plt.subplot(1, 2, 2)
        plt.title("Measurement function $Im f(x,\lambda^*)$")
        plt.plot(xs, fs[:, 1], linewidth=5)
        plt.scatter(xs_meas, ys_meas[:, 1], alpha=0.5, c="lightblue", s=10)
        plt.xlabel("$x$")
        plt.ylabel("$Imaginary f(x,\lambda^*)$")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Measurement function $|f(x,\lambda^*)|$")
        plt.plot(xs, np.abs(fs[:, 0] + 1j*fs[:, 1]), linewidth=5)
        plt.xlabel("$x$")
        plt.ylabel("$|f(x,\lambda^*)|$")
        plt.ylim([0, None])

        plt.subplot(1, 2, 2)
        plt.title("Measurement function $Arg(f(x,\lambda^*))$")
        plt.plot(xs, np.angle(fs[:, 0] + 1j*fs[:, 1]), linewidth=5)
        plt.xlabel("$x$")
        plt.ylabel("$Arg f(x,\lambda^*)$")
        plt.ylim([-np.pi - 0.05, np.pi + 0.05])
        plt.show()

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["real_lambda"] = self.tf_real_lambda.numpy()
        h5_handle.attrs["x_range"] = self.x_range
        self.p.save(h5_handle.create_group("p"))

    def __repr__(self):
        dict_str = ""
        for key, val in self.__dict__.items():
            if type(val) != Cavities.Parameters:
                dict_str += f"{key:25}:\t{val}\n"

        return f"""{self.__class__}
{dict_str}
Parameters:
{self.p.__repr__()}
    """


class LinearToy(System):
    """
    Simplest example, which is Gaussian, useful for testing.

    It implements the response function f(x) = λ_1 cos x + λ_2 sin x.
    """
    @dataclass
    class Parameters:
        sigma_noise: float

        def __repr__(self) -> str:
            repr_str = ""
            for key, val in self.__dict__.items():
                repr_str += f"{key:25}:\t{val}\n"
            return repr_str

        def save(self, h5_handle):
            return al.utils.h5_save_dataclass(self, h5_handle.attrs)

    dim_x = 1
    dim_y = 1
    tf_real_lambda = np.array([])

    def __init__(self,
                 real_lambda=[],
                 sigma_noise=0.3,
                 x_range=None,
                 **kwargs):

        self.p = self.Parameters(sigma_noise=sigma_noise)

        if real_lambda == []:
            real_lambda = [0.5, 1.]
        self.tf_real_lambda = tf.convert_to_tensor(real_lambda, K.backend.floatx())
        self.x_range = x_range

    @classmethod
    def from_h5_handle(cls, h5_handle):
        real_lambda = h5_handle.attrs["real_lambda"]
        x_range = list(h5_handle.attrs["x_range"])  # needed for plots
        return cls(**dict(h5_handle["p"].attrs), real_lambda=real_lambda, x_range=x_range)

    def measurement_curve(self, learner, label):
        return measurement_curve_1d(learner, label)

    def tf_f(self, tf_x, tf_lambda):
        return (tf.cos(tf_x)*tf_lambda[..., 0] + tf.sin(tf_x)*tf_lambda[..., 1])[..., None]

    def measure(self, x):
        f = self.tf_f(x, self.tf_real_lambda).numpy()
        noise = np.random.normal(size=f.shape)*self.p.sigma_noise
        return f + noise

    def plot_response(self, PLOT_EXTENT=[-np.pi, np.pi]):
        xs = np.linspace(*PLOT_EXTENT, 150)
        tf_xs = tf.convert_to_tensor(xs, K.backend.floatx())
        fs = self.tf_f(tf_xs, self.tf_real_lambda).numpy()

        xs_meas = np.repeat(xs, 5)
        tf_xs_meas = tf.convert_to_tensor(xs_meas, K.backend.floatx())
        ys_meas = self.measure(tf_xs_meas)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Measurement function $f(x,\lambda^*)$")
        plt.plot(xs, fs, linewidth=5)
        plt.scatter(xs_meas, ys_meas, alpha=0.5, c="lightblue", s=10)
        plt.xlabel("$x$")
        plt.ylabel("$f(x,\lambda^*)$")
        plt.show()

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["real_lambda"] = self.tf_real_lambda.numpy()
        h5_handle.attrs["x_range"] = self.x_range
        self.p.save(h5_handle.create_group("p"))

    def __repr__(self):
        dict_str = ""
        for key, val in self.__dict__.items():
            if type(val) != LinearToy.Parameters:
                dict_str += f"{key:25}:\t{val}\n"

        return f"""{self.__class__}
{dict_str}
Parameters:
{self.p.__repr__()}
    """


class Qubits(System):

    @dataclass
    class Parameters:
        j_coupling: float
        alpha_pulse: float

        def __repr__(self) -> str:
            repr_str = ""
            for key, val in self.__dict__.items():
                repr_str += f"{key:25}:\t{val}\n"
            return repr_str

        def save(self, h5_handle):
            return al.utils.h5_save_dataclass(self, h5_handle.attrs)

    dim_x = 1
    dim_y = 1
    tf_real_lambda = np.array([])

    def __init__(self,
                 real_lambda=[],
                 j_coupling=0.5,
                 alpha_pulse=np.pi,
                 dim_lambda=None, type_lambda=None,
                 x_range=None, **kwargs):

        self.p = self.Parameters(
            j_coupling=j_coupling,
            alpha_pulse=alpha_pulse
        )

        if real_lambda != []:
            real_lambda = np.array(real_lambda)
            if dim_lambda is not None and type_lambda is not None:
                al.logger.warning("real_lambda specified. Ignoring dim_lambda and type_lambda parameters")

        if dim_lambda is not None and type_lambda is not None:
            if type_lambda == "all_ones":
                real_lambda = np.ones(dim_lambda)
            elif type_lambda == "gaussian_random":
                real_lambda = np.random.randn(dim_lambda)
            else:
                raise Exception("What do you want?")
        else:
            if dim_lambda is not None and type_lambda is not None:
                if type_lambda == "all_ones":
                    real_lambda = np.ones(dim_lambda)
                elif type_lambda == "gaussian_random":
                    real_lambda = np.random.randn(dim_lambda)
                else:
                    raise Exception("What do you want?")

        # if lambda are the frequencies, then lambda has the size of the system (size of H_mat)
        self.tf_real_lambda = tf.convert_to_tensor(real_lambda, K.backend.floatx())
        self.dim_system = self.dim_lambda
        self.x_range = x_range  # x range list for plots

        # Hamiltonian parameters
        self.frequencies = [1.]*self.dim_system

        # Excited pulse
        self.excited_qubit_idx = 0

        # Measured qubit
        self.measurement_idx = 0

        self.sigma_x = tf.constant([[0, 1], [1, 0]], dtype=tf_complex)
        self.sigma_y = tf.constant([[0j, -1j], [1j, 0j]], dtype=tf_complex)
        self.sigma_z = tf.constant([[1, 0], [0, -1]], dtype=tf_complex)

        self.tf_n_qubits = tf.convert_to_tensor(self.dim_system, tf_complex)
        self.tf_frequencies = tf.convert_to_tensor(self.frequencies, tf_complex)

        self.tf_coupling = tf.convert_to_tensor(self.p.j_coupling, tf_complex)
        self.tf_alpha_pulse = tf.convert_to_tensor(self.p.alpha_pulse, tf_complex)

    @classmethod
    def from_h5_handle(cls, h5_handle):
        real_lambda = h5_handle.attrs["real_lambda"]
        x_range = list(h5_handle.attrs["x_range"])  # needed for plots
        return cls(**dict(h5_handle["p"].attrs), real_lambda=real_lambda, x_range=x_range)

    def measurement_curve(self, learner, label):
        return measurement_curve_1d(learner, label, only_one_true_line=True)

    def tf_kron(self, tf_A, tf_B):
        tf_shape = tf_A.shape[-1]*tf_B.shape[-1]
        return tf.reshape(tf_A[..., :, None, :, None]*tf_B[..., None, :, None, :], (-1, tf_shape, tf_shape))

    def tf_get_H(self, tf_frequencies, tf_coupling):
        tf_H_diag = sum(self.tf_kron(
            self.tf_kron(tf.eye(2**i, dtype=tf_complex), tf_frequencies[..., i, None, None]*self.sigma_z),
            tf.eye(2**(self.dim_system-i-1), dtype=tf_complex))
            for i in range(self.dim_system))

        tf_H_int = tf_coupling*sum(self.tf_kron(
            self.tf_kron(
                self.tf_kron(self.tf_kron(tf.eye(2**i, dtype=tf_complex), self.sigma_x),
                             tf.eye(2**(j-i-1), dtype=tf_complex)),
                self.sigma_x),
            tf.eye(2**(self.dim_system-j-1), dtype=tf_complex)) for i in range(self.dim_system-1) for j in range(i+1, i+2))

        return tf_H_diag + tf_H_int

    def tf_ground_state(self, tf_H):
        tf_eigvals, tf_eigvects = tf.eig(tf_H)
        tf_indices = tf.argmin(tf.math.real(tf_eigvals), axis=-1)
        tf_ground = tf.gather(tf_eigvects, tf_indices, axis=-1, batch_dims=1)
        return tf_ground

    def tf_apply_pulse(self, tf_alpha_pulse, psi):
        U_pulse_ = tf.linalg.expm(-1j*tf_alpha_pulse*self.sigma_x/2)
        U_pulse = self.tf_kron(
            self.tf_kron(tf.eye(2**self.excited_qubit_idx, dtype=tf_complex), U_pulse_),
            tf.eye(2**(self.dim_system-self.excited_qubit_idx-1), dtype=tf_complex))[0]
        tf_excited = tf.tensordot(psi, U_pulse, axes=[-1, -1])
        return tf_excited

    def tf_evolve_psi(self, tf_H, tf_psi, tf_dt):
        U_evolution_ = tf.linalg.expm(-1j*tf_H*tf_dt)   # Change evolution matrix
        tf_evolved = tf.einsum('...ij,...j->...i', U_evolution_, tf_psi)
        return tf_evolved

    def tf_density_mat(self, tf_psi):
        return tf.einsum('...i,...k->...ik', tf.math.conj(tf_psi), tf_psi)

    def tf_measure_probs(self, tf_rho):
        qubit_reshape = [2**self.measurement_idx, 2, 2**(self.dim_system-self.measurement_idx-1)]
        tf_rho_qubits = tf.reshape(tf_rho, [-1, *(qubit_reshape*2)])

        tf_rho_traced = tf.linalg.trace(tf.linalg.trace(tf.transpose(tf_rho_qubits, (0, 2, 5, 1, 4, 3, 6))))
        tf_probs = tf.linalg.diag_part(tf.einsum('ij,...jk,kl->...il', self.sigma_z, tf_rho_traced, self.sigma_z))

        return tf_probs

    def tf_get_probs(self, tf_psi_evolved):
        qubit_reshape = [2]*self.dim_lambda
        tf_psi_evolved_qubits = tf.reshape(tf.transpose(tf_psi_evolved), (*qubit_reshape, -1))  # -1 is batch_size

        sum_indices = tuple(range(self.measurement_idx))+tuple(range(self.measurement_idx+1, self.dim_lambda))
        tf_probs = tf.transpose(  # transpose to put batch_size first
            tf.reduce_sum(tf.abs(tf_psi_evolved_qubits)**2, sum_indices)
        )
        return tf_probs

    # @tf.function
    def tf_f(self, tf_x, tf_lambda):
        tf_x_ = tf.cast(tf.complex(tf_x, 0.0), tf_complex)  # represents evolution time
        tf_lambda_ = tf.cast(tf.complex(tf_lambda, 0.0), tf_complex)  # represents qubit frequencies

        tf_H = self.tf_get_H(tf_lambda_, self.tf_coupling)
        tf_psi_ground = self.tf_ground_state(tf_H)
        tf_psi_excited = self.tf_apply_pulse(self.tf_alpha_pulse, tf_psi_ground)
        tf_psi_evolved = self.tf_evolve_psi(tf_H, tf_psi_excited, tf_x_)

        tf_probs = self.tf_get_probs(tf_psi_evolved)

        # Useful test cases
        # np.testing.assert_almost_equal(tf.linalg.norm(tf_psi_ground[0]).numpy(),1., decimal=3)
        # np.testing.assert_almost_equal(tf.linalg.norm(tf_psi_excited[0]).numpy(),1., decimal=3)
        # np.testing.assert_almost_equal(tf.linalg.norm(tf_psi_evolved[0]).numpy(),1.,decimal=3)
        # np.testing.assert_almost_equal(tf.reduce_sum(tf_probs[0]),1.,decimal=3)

        return tf.cast(tf.math.real(tf_probs), K.backend.floatx())

    def tf_fisher_function_plain(self, tf_x, tf_lambda):
        ...

    def measure(self, x):
        tf_x = tf.cast(x, K.backend.floatx())
        f = self.tf_f(tf_x, self.tf_real_lambda).numpy()
        # Todo: test sum of probabilities
        return np.random.choice([0, 1], p=[f[0, 0], 1-f[0, 0]])[None]

    def plot_response(self, PLOT_EXTENT=None):
        if PLOT_EXTENT is None:
            PLOT_EXTENT = al.config.cfg.SYSTEM.X_RANGE
        xs = np.linspace(*PLOT_EXTENT, 100)
        fs = []
        for x in xs:
            tf_x = tf.convert_to_tensor(x, K.backend.floatx())
            fs.append(self.tf_f(tf_x, self.tf_real_lambda).numpy().squeeze())

        xs_meas = np.repeat(xs, 1)
        ys_meas = np.array([self.measure(x) for x in xs_meas])

        # Convert to resemble probability shape (i.e. prob=1 -> measurement 1 (instead of 0))
        # This makes it easier to check if they overlap when plotting
        ys_meas = 1 - np.clip(ys_meas, 0, None)

        plt.figure(figsize=(10, 5))
        plt.title("Measurement function $Re f(x,\lambda^*)$")
        plt.plot(xs, np.array(fs)[:, 1], linewidth=5)
        plt.scatter(xs_meas, ys_meas, alpha=0.5, c="green", s=10)
        plt.xlabel("$x$")
        plt.ylabel("$Real f(x,\lambda^*)$")
        plt.show()

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["real_lambda"] = self.tf_real_lambda.numpy()
        h5_handle.attrs["x_range"] = self.x_range
        self.p.save(h5_handle.create_group("p"))

    def __repr__(self):
        dict_str = ""
        for key, val in self.__dict__.items():
            if type(val) != Qubits.Parameters:
                dict_str += f"{key:25}:\t{val}\n"

        return f"""{self.__class__}
{dict_str}
Parameters:
{self.p.__repr__()}
    """


class QubitsBinomial(Qubits):
    """
    Qubit chain system, but performing multiple measurements with the same settings to increase statistics.
    """

    def __init__(self,
                 real_lambda=[],
                 j_coupling=0.5,
                 alpha_pulse=np.pi,
                 dim_lambda=None,
                 type_lambda=None,
                 x_range=None,
                 n_counts=100,
                 **kwargs):
        super().__init__(real_lambda, j_coupling, alpha_pulse, dim_lambda, type_lambda, x_range, **kwargs)
        self.n_counts = n_counts

    @classmethod
    def from_h5_handle(cls, h5_handle):
        real_lambda = h5_handle.attrs["real_lambda"]
        x_range = list(h5_handle.attrs["x_range"])  # needed for plots
        n_counts = h5_handle.attrs["n_counts"]

        return cls(**dict(h5_handle["p"].attrs), real_lambda=real_lambda, x_range=x_range, n_counts=n_counts)

    def save(self, h5_handle):
        h5_handle.attrs[al.utils.CLASS_NAME_PROPERTY] = type(self).__name__
        h5_handle.attrs["real_lambda"] = self.tf_real_lambda.numpy()
        h5_handle.attrs["x_range"] = self.x_range
        h5_handle.attrs["n_counts"] = self.n_counts

        self.p.save(h5_handle.create_group("p"))

    def measure(self, x):
        tf_x = tf.cast(x, K.backend.floatx())
        f = self.tf_f(tf_x, self.tf_real_lambda).numpy()
        # Todo: ! test sum of probabilities
        return np.random.binomial(n=self.n_counts, p=f[0, 0], size=(*x.shape, 1))/self.n_counts


def get_system_from_name(name):
    if name == Cavities.__name__:
        return Cavities
    elif name == LinearToy.__name__:
        return LinearToy
    elif name == Qubits.__name__:
        return Qubits
    elif name == QubitsBinomial.__name__:
        return QubitsBinomial


# Some plotting functions:
def measurement_curve_2d(learner, label="experiment"):
    xs = np.linspace(*learner.system.x_range, 150)
    delta_xs = xs[1] - xs[0]

    batch_size = 10000
    proposed_lambdas = learner.prior.sample(batch_size)

    # Show measurement plot
    mean_lambda = tf.reduce_mean((proposed_lambdas), 0)
    std_lambda = tf.math.reduce_std((proposed_lambdas), 0)

    y_linsp = np.array([learner.system.tf_f(tf.convert_to_tensor(xxx, K.backend.floatx()),
                                            learner.system.tf_real_lambda).numpy() for xxx in xs])
    y_linsp_approx = np.array([learner.system.tf_f(tf.convert_to_tensor(xxx, K.backend.floatx()),
                                                   proposed_lambdas[:100, :]).numpy() for xxx in xs])

    plt.figure(figsize=[13, 9])
    plt.suptitle(f"#{label} - Measurement function prediction")

    plt.subplot(2, 2, 1)
    plt.title("Real $f(x,\lambda^*)$")
    plt.plot(xs, y_linsp[:, 0], label="True")
    plt.plot(xs, y_linsp_approx[:, :, 0], c="orange", alpha=0.05)
    plt.xlabel("$x$")
    plt.ylabel("$f(x,\lambda^*)$")
    plt.xlim(*learner.system.x_range)

    N_points = len(learner.history.x)
    if N_points > 0:
        colors = plt.cm.Blues(np.linspace(0, 1, N_points + 1))
        plt.scatter(learner.history.x, np.array(learner.history.y)[:, 0], color=colors[1:])
        for i in range(N_points):
            plt.annotate(i,
                         (learner.history.x[i] + 0.05, np.array(learner.history.y)[i, 0] + 0.05))
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Imaginary $f(x,\lambda^*)$")
    plt.plot(xs, y_linsp[:, 1], label="True")
    plt.plot(xs, y_linsp_approx[:, :, 1], c="orange", alpha=0.05)
    plt.xlabel("$x$")
    plt.ylabel("$f(x,\lambda^*)$")
    plt.xlim(*learner.system.x_range)

    if N_points > 0:
        colors = plt.cm.Blues(np.linspace(0, 1, N_points + 1))
        plt.scatter(learner.history.x, np.array(learner.history.y)[:, 1], color=colors[1:])
        for i in range(N_points):
            plt.annotate(i,
                         (learner.history.x[i] + 0.05, np.array(learner.history.y)[i, 1] + 0.05))
    plt.legend()

    if al.metrics.ExpectedInformationGainCurve.__name__ in learner.metrics.keys():
        plt.subplot(2, 2, 3)

        #[n_measurement, x_vals, (x or ig)]
        ordered_metric = np.array(
            learner.metrics[al.metrics.ExpectedInformationGainCurve.__name__].history
        )[..., 1]

        plt.imshow((np.array(ordered_metric).T).T,
                   extent=(*learner.system.x_range, 0, len(learner.history.x)), origin="lower", aspect="auto")

        # plt.imshow((np.array(ordered_metric).T / np.max(ordered_metric, 1)).T,
        #     extent=(*learner.system.x_range, 0, len(learner.history.x)), origin="lower", aspect="auto")

        plt.scatter(learner.history.x, np.arange(len(learner.history.x))+0.5, c="red")
        plt.title("Information Gain (expected at each step)")
        plt.xlabel("measured x")
        plt.ylabel("# measurement")
        plt.xlim(*learner.system.x_range)

        plt.subplot(2, 2, 4)
        plt.imshow((np.array(ordered_metric).T / np.max(ordered_metric, 1)).T,
                   extent=(*learner.system.x_range, 0, len(learner.history.x)), origin="lower", aspect="auto")
        plt.scatter(learner.history.x, np.arange(len(learner.history.x))+0.5, c="red")
        plt.title("Information Gain (expected at each step)")
        plt.xlabel("measured x")
        plt.ylabel("# measurement")
        plt.xlim(*learner.system.x_range)

    plt.savefig(f"{al.config.Directories().PATH_OUTPUT_FIGURES}/measurements-and-ig_{label}.pdf")
    plt.show()


# TODO: refactor. WHY DO WE HAVE TWO functions for the same thing?
def measurement_curve_1d(learner, label="experiment", only_one_true_line=False):
    # only_one_true_line: plot only one component of y_linsp_approx
    # (for example, p(0) for qubits systems)
    # it does not need to plot p(1)=1-p(0))
    # This function will be called by the learner
    # (inheriting the plotting logic)
    xs = np.linspace(*learner.system.x_range, 150)
    delta_xs = xs[1] - xs[0]

    batch_size = 10000
    proposed_lambdas = learner.prior.sample(batch_size)

    # Show measurement plot
    mean_lambda = tf.reduce_mean((proposed_lambdas), 0)
    std_lambda = tf.math.reduce_std((proposed_lambdas), 0)

    y_linsp = np.array([learner.system.tf_f(tf.convert_to_tensor(xxx, K.backend.floatx()),
                                            learner.system.tf_real_lambda).numpy() for xxx in xs])
    y_linsp_approx = np.array([learner.system.tf_f(tf.convert_to_tensor(xxx, K.backend.floatx()),
                                                   proposed_lambdas[:100, :]).numpy() for xxx in xs])

    plt.figure(figsize=[13, 7])
    plt.suptitle(f"#{label} - Measurement function prediction")

    plt.subplot(2, 2, 1)
    plt.title("Real $f(x,\lambda^*)$")
    if only_one_true_line:
        plt.plot(xs,  y_linsp[:, 0, 0], label="True")
    else:
        plt.plot(xs, y_linsp[:, 0], label="True")

    plt.plot(xs, y_linsp_approx[:, :, 0], c="orange", alpha=0.05)

    plt.xlabel("$x$")
    plt.ylabel("$f(x,\lambda^*)$")
    plt.xlim(*learner.system.x_range)

    N_points = len(learner.history.x)
    if N_points > 0:
        colors = plt.cm.Blues(np.linspace(0, 1, N_points + 1))
        ys_meas = np.array(learner.history.y)[:, 0]
        plt.scatter(learner.history.x, ys_meas, color=colors[1:])
        for i in range(N_points):
            plt.annotate(i,
                         (learner.history.x[i] + 0.05, ys_meas[i] + 0.05))
    plt.legend()

    if al.metrics.ExpectedInformationGainCurve.__name__ in learner.metrics.keys():
        plt.subplot(2, 2, 3)

        #[n_measurement, x_vals, (x or ig)]
        ordered_metric = np.array(
            learner.metrics[al.metrics.ExpectedInformationGainCurve.__name__].history
        )[..., 1]

        plt.imshow((np.array(ordered_metric).T).T,
                   extent=(*learner.system.x_range, 0, len(learner.history.x)), origin="lower", aspect="auto")
        plt.clim([-1, None])
        # plt.imshow((np.array(ordered_metric).T / np.max(ordered_metric, 1)).T,
        #         extent=(*learner.system.x_range, 0, len(learner.history.x)), origin="lower", aspect="auto")

        plt.scatter(learner.history.x, np.arange(len(learner.history.x))+0.5, c="red")
        plt.title("Information Gain (expected at each step)")
        plt.xlabel("measured x")
        plt.ylabel("# measurement")
        plt.xlim(*learner.system.x_range)

    plt.savefig(f"{al.config.Directories().PATH_OUTPUT_FIGURES}/measurements-and-ig_{label}.png")
    plt.show()
