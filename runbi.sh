#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gpus=2
#SBATCH --time=1-2:34:56
#SBATCH --account=kaiechen
#SBATCH --mail-type=all
#SBATCH --mail-user=kaiechen@ucsb.edu
#SBATCH --output=stdoutbi.txt
#SBATCH --error=stderr.txt

echo "JOB ID: $SLURM_JOB_ID"
echo "JOB USER: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

srun python3 finetunebi.py
