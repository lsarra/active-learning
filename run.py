#!/usr/bin/env python
import papermill as pm
import activelearning as al
from activelearning.config import cfg, Directories
import sys
import argparse
import uuid


class args:
    notebook_path = "notebook.ipynb"
    config_path = ""


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Active Learning Experiment",
    epilog="A folder with experiment name will be created in /outputs",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-notebook_path", type=str, default=args.notebook_path)
parser.add_argument("-config_path", type=str, default=args.config_path)

try:
    args = parser.parse_args()
except SystemExit as e:
    print("Running from interactive session. Loading default parameters")

if args.config_path != "":
    cfg.merge_from_file(args.config_path)

experiment_output = f"{Directories().PATH_OUTPUT_NOTEBOOKS}/{cfg.PATH.SUBLABEL}.ipynb"
al.utils.init_output_directories()

pm.execute_notebook(
    args.notebook_path,
    experiment_output,
    parameters={
        "config_path": f"{Directories().PATH_OUTPUT_CONFIGS}/{cfg.PATH.SUBLABEL}.yaml"
    },
)