#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=2:34:56
#SBATCH --account=tianruigu
#SBATCH --mail-type=all
#SBATCH --mail-user=tianruigu@ucsb.edu 
#SBATCH --output=stdout_eval.txt
#SBATCH --error=stderr_eval.txt


echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


srun python3 NLLB_new.py --model_path ./model-1
echo "== AVERGAE" >> result_finetune.txt
bash evaluate.sh

srun python3 NLLB_new.py --model_path ./model-2
echo "== AVERGAE" >> result_finetune.txt
bash evaluate.sh