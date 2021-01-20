#SBATCH --qos=dque
#SBATCH --mem-per-cpu=16GB 
#SBATCH --time=01-00:00:00

id=$SLURM_JOBID"_"$SLURM_ARRAY_TASK_ID
i=$SLURM_ARRAY_TASK_ID
params=$1
echo "id:"
echo $id

source activate audneurorl
module add cuda

echo "---------------------"
echo "launching run "$i

cd /fs/clip-realspeech/projects/vid_game/software/VidGameRL
python create_params_file.py ../params/$id".params" "$params" -run_num=$i

echo "param file created"
echo "starting training"
python main_game_conv.py ../params/$id".params"
echo "training complete"

echo "starting testing"
python test_conv.py ../params/$id".params"

echo "finished"