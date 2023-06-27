# This is the default configuration
from yacs.config import CfgNode as ConfigurationNode

C = ConfigurationNode()

C.PATH = ConfigurationNode()
C.PATH.OUTPUT_NAME = "outputs"
C.PATH.EXPERIMENT_NAME = "default_experiment"
C.PATH.SUBLABEL = "default"  # sublabels will be analyzed together


C.PATH.LOGS_NAME = "logs"
C.PATH.MODELS_NAME = "models"
C.PATH.FIGURES_NAME = "figures"
C.PATH.NOTEBOOKS_NAME = "notebooks"
C.PATH.CONFIGS_NAME = "configs"

C.PARALLEL = ConfigurationNode()
C.PARALLEL.ENABLED = True
C.PARALLEL.N_NODES = 10
C.PARALLEL.PROFILE_ID = "parallel_jpt"
C.PARALLEL.CLUSTER_ID = "jupyter"
C.PARALLEL.LOG = False  # should be true only on a worker (i.e. parallel node)

C.SYSTEM = ConfigurationNode()
C.SYSTEM.TYPE = "Cavities"  # LinearToy Qubits
C.SYSTEM.DIM_LAMBDA = 2
C.SYSTEM.TYPE_LAMBDA = "gaussian_random"
C.SYSTEM.REAL_LAMBDA = []  # if specified, uses these parameters instead of sampling

# Cavities
C.SYSTEM.SIGMA_NOISE = 0.05
C.SYSTEM.SIZE = 3
# Qubits
C.SYSTEM.COUPLING = 7.0
C.SYSTEM.COUPLING_RANDOM = False

C.SYSTEM.X_RANGE = [-6, 6]

# In case of Binomial likelihood (for the Qubits system):
C.SYSTEM.BINOMIAL = ConfigurationNode()
C.SYSTEM.BINOMIAL.N_COUNTS = 50


C.LEARNER = ConfigurationNode()
C.LEARNER.DISCRETE = ConfigurationNode()
C.LEARNER.DISCRETE.ENABLED = True
C.LEARNER.DISCRETE.X_RANGE = [-6, 6, 55]
C.LEARNER.DISCRETE.Y1_RANGE = [-1.2, 1.2, 54]
C.LEARNER.DISCRETE.Y2_RANGE = [-1.2, 1.2, 54]
C.LEARNER.DISCRETE.LAMBDA1_RANGE = [-3.5, 3.5, 45]
C.LEARNER.DISCRETE.LAMBDA2_RANGE = [-3.5, 3.5, 45]


C.LEARNER.POSTERIOR = ConfigurationNode()
# NormalizingFlow Gaussian
C.LEARNER.POSTERIOR.TYPE = "NormalizingFlow"


C.ADVISOR = ConfigurationNode()
# InformationGainAdvisor RandomAdvisor FixedAdvisor
C.ADVISOR.NAME = "InformationGainAdvisor"
C.ADVISOR.N_TRAIN = 6000
C.ADVISOR.BATCH_SIZE = 500

C.TRAINING = ConfigurationNode()
C.TRAINING.N_MEASUREMENTS = 10

C.METRICS = ConfigurationNode()
C.METRICS.EXP_IG = ConfigurationNode()
C.METRICS.EXP_IG.N_TRAIN = 3000

cfg = C


class Directories:
    def __init__(self):
        self.PATH_OUTPUT_EXPERIMENT = (
            cfg.PATH.OUTPUT_NAME + "/" + cfg.PATH.EXPERIMENT_NAME
        )
        self.PATH_OUTPUT_LOGS = self.PATH_OUTPUT_EXPERIMENT + "/" + cfg.PATH.LOGS_NAME
        self.PATH_OUTPUT_MODELS = (
            self.PATH_OUTPUT_EXPERIMENT + "/" + cfg.PATH.MODELS_NAME
        )
        self.PATH_OUTPUT_FIGURES = (
            self.PATH_OUTPUT_EXPERIMENT + "/" + cfg.PATH.FIGURES_NAME
        )
        self.PATH_OUTPUT_NOTEBOOKS = (
            self.PATH_OUTPUT_EXPERIMENT + "/" + cfg.PATH.NOTEBOOKS_NAME
        )
        self.PATH_OUTPUT_CONFIGS = (
            self.PATH_OUTPUT_EXPERIMENT + "/" + cfg.PATH.CONFIGS_NAME
        )

    def __repr__(self) -> str:
        repr_str = ""
        for key, val in self.__dict__.items():
            repr_str += f"{key:25}:\t{val}\n"
        return repr_str
