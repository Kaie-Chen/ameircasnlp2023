#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --account=tianruigu
#SBATCH --mail-type=all
#SBATCH --mail-user=tianruigu@ucsb.edu 
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt


echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


srun --cpus-per-task=4 --mem-per-cpu=4GB --gpus=1 --time=1-2:34:56 --pty python3 NLLB_new.py
