#! /bin/bash -l
#
# Standard output and error: [filename pattern]
#SBATCH -o ./logs/current.ipcontroller.log
#SBATCH -e ./logs/current.ipcontroller-err.log
#
# Initial working directory:
#SBATCH -D ./
#
# Job Name:
#SBATCH -J active_ctrl
#
# Queue (Partition):
#SBATCH --partition=standard,gpu
#
# Process management (number of parallel executions is specified using the --array option):
#     * possible formats: `--array=0-9`, `--array=1,3,5,7`, `--array=1-7:2`
#     * reduce maximum number of simultaneously running tasks using a "%" separator (e.g. `--array=0-9%4`)
#     * to start only one instance, use --array=0 or (better) leave the --array option away completely
##SBATCH --array=0
##SBATCH --gres=gpu:1           # specify number of GPUs
##SBATCH --nodes=5
##SBATCH --ntasks-per-node=8     # or use directly ntasks

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # specify number of CPU cores (as a rule of thumb, 4 per GPU)

#
# Memory requirement (default is 64GB):
#SBATCH --mem=64GB
#
# Wall clock limit:
#SBATCH --time=120:10:00

source load_conda
module load intel/21.2.0
module load impi/2021.2 
srun ipcontroller --profile-dir=/zeropoint/u/lsarra/.ipython/profile_parallel_jpt --cluster-id=jupyter --location=$HOSTNAME
