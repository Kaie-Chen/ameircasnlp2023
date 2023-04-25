#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gpus=2
#SBATCH --time=1-2:34:56
#SBATCH --account=tianruigu
#SBATCH --mail-type=all
#SBATCH --mail-user=tianruigu@ucsb.edu 
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt


echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


srun python3 finetune.py
