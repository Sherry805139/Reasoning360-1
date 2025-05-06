#!/bin/bash
#SBATCH --job-name=ZebraLogicDataGen
#SBATCH --nodes=1   
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Request 8 CPUs for your threads
#SBATCH --time=4-00:00:00       # Adjust runtime as needed
#SBATCH --gres=gpu:1              


# Load any required modules
module load conda
conda activate base

# Set Python to use the specified number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run your Python script
python puzzle_generator.py --num_puzzles 10000 --num_processes 32