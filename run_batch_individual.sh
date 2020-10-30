#SBATCH --qos=dque
#SBATCH --mem-per-cpu=16GB 
#SBATCH --time=01-00:00:00

i=$SLURM_ARRAY_TASK_ID
overwrite=$1

source activate audneurorl

echo "---------------------"
echo "launching run "$i
python main_game.py -modelname=run"$i"
echo "finished"