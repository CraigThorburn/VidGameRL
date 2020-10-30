#!/bin/bash
#SBATCH --qos=dque
#SBATCH --mem-per-cpu=8GB 
#SBATCH --time=01-00:00:00

i=$SLURM_ARRAY_TASK_ID
overwrite=$1

conda activate audneurorl

echo "---------------------"
echo "launching run "$i
python main_game.py $host -overwrite $overwrite -modelname run$i
echo "finished"