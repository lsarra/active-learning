import logging
import sys
import os
import activelearning as al
from activelearning.config import cfg
import h5py
import tensorflow as tf

CLASS_NAME_PROPERTY = "class_name"


def init(config_path=None):
    tf_allocate_gpu()
    if config_path is not None:
        cfg.merge_from_file(config_path)
    al.utils.init_output_directories()
    al.utils.init_logfile(f"{al.config.Directories().PATH_OUTPUT_LOGS}/{cfg.PATH.SUBLABEL}.log")


def init_logfile(LOG_PATH="logs/last_run.log"):
    logger = logging.getLogger("active-learning")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d-%(funcName)s %(message)s')

    fh = logging.FileHandler(f'{LOG_PATH}')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f"Running job #{os.environ.get('SLURM_JOB_ID')}")
    logger.info(f"Log file has been created in {LOG_PATH}")


def init_output_directories():
    make_dir("logs")
    make_dir(cfg.PATH.OUTPUT_NAME)
    make_dir(al.config.Directories().PATH_OUTPUT_EXPERIMENT)
    make_dir(al.config.Directories().PATH_OUTPUT_LOGS)
    make_dir(al.config.Directories().PATH_OUTPUT_MODELS)
    make_dir(al.config.Directories().PATH_OUTPUT_FIGURES)
    make_dir(al.config.Directories().PATH_OUTPUT_NOTEBOOKS)
    make_dir(al.config.Directories().PATH_OUTPUT_CONFIGS)
    cfg.dump(stream=open(f"{al.config.Directories().PATH_OUTPUT_CONFIGS}/{cfg.PATH.SUBLABEL}.yaml", "w"))
    print("Creating directories...")


def tf_allocate_gpu():
    """
    This function solves possible memory allocation issues in TensorFlow.
    """

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def h5_save_dataclass(dataclass, h5_handle):
    for k in dataclass.__dataclass_fields__:
        h5_handle[k] = getattr(dataclass, k)


def h5_load_dataclass(dataclass, h5_handle):
    for k in h5_handle.keys():
        setattr(dataclass, k, h5_handle[k][:])


def h5_print_attrs(name, obj):
    """"
    Prints the attributes of a hdf5 object.
    """
    ##usage: file.visititems(h5_print_attrs)
    # Create indent
    shift = name.count('/') * '    '
    item_name = name.split("/")[-1]
    print(shift + item_name)
    try:
        for key, val in obj.attrs.items():
            print(shift + '    ' + f"{key}: {val}")
    except:
        pass


def h5_get_model_path(label):
    """
    Gets the model hdf5 object path given its ID.
    """
    if not label.endswith(".hdf5"):
        label += ".hdf5"
    return f"{al.config.Directories().PATH_OUTPUT_MODELS}/{label}"


def save_all(saving_path, system, learner, advisor):
    """"
    Checkpoints a measurement step.
    """
    with h5py.File(saving_path, mode="w") as f:
        system.save(f.create_group("system"))
        learner.save(f.create_group("learner"))
        advisor.save(f.create_group("advisor"))
        advisor.last_save_path = saving_path
    al.logger.info(f"Model saved as {saving_path}.")


def load_all(loading_path, parallel=None, load_discrete=False):
    """
    Loads a model checkpoint. 

    load_discrete allows to load the discrete calculations (used for testing)
    """
    with h5py.File(loading_path, mode="r") as file:
        system_name = file["system"].attrs[al.utils.CLASS_NAME_PROPERTY]
        system = al.systems.get_system_from_name(system_name).from_h5_handle(file["system"])

        learner = al.learners.BayesLearner.from_h5_handle(system, file["learner"],
                                                          load_discrete=load_discrete)

        advisor_name = file["advisor"].attrs[al.utils.CLASS_NAME_PROPERTY]
        advisor = al.advisors.get_advisor_from_name(advisor_name).from_h5_handle(
            learner, file["advisor"], parallel)
        advisor.last_save_path = loading_path
        learner.metrics.load_all(file["learner/metrics"])

    al.logger.info(f"Loaded model from {loading_path}.")
    return system, learner, advisor


def create_new_cell(contents):
    from IPython.core.getipython import get_ipython
    shell = get_ipython()

    payload = dict(
        source='set_next_input',
        text=contents,
        replace=False,
    )
    shell.payload_manager.write_payload(payload, single=False)


def save_nn_weights(net_weights, h5_handle):
    """Dumps the weights of a neural network in a hdf5 leaf."""
    for i, lst in enumerate(net_weights):
        h5_handle.create_dataset(str(i), data=lst)


def load_nn_weights(h5_handle):
    """Loads the weights of a neural network from a hdf5 leaf."""

    new_weights = []
    for key in h5_handle.keys():
        new_weights.append(h5_handle[key][:])
    return new_weights
