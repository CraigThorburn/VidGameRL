#SBATCH --qos=dque
#SBATCH --mem-per-cpu=16GB 
#SBATCH --time=01-00:00:00

id=$SLURM_JOBID"_"$SLURM_ARRAY_TASK_ID
i=$SLURM_ARRAY_TASK_ID
input="$@"
stage="${input[@]:0:1}"
params="${input[@]:1}"
echo "id:"
echo $id
echo 'params: '
echo $params
echo 'stage: '
echo $stage
source activate audneurorl
module add cuda

echo "---------------------"
echo "launching run "$i

cd /fs/clip-realspeech/projects/vid_game/software/VidGameRL
python create_params_file.py ../params/$id".params" $params -run_num=$i
echo "param file created"

if [ $stage -le 0 ]; then
echo "starting training"
python main_game_conv.py ../params/$id".params"
echo "training complete"
fi

if [ $stage -le 1 ]; then
echo "starting training results processing"
python process_game_experiment_from_file.py ../params/$id".params false"
echo "training results processing complete"
fi

if [ $stage -le 2 ]; then
echo "starting testing"
python test_conv.py ../params/$id".params"
echo "testing complete"
fi

if [ $stage -le 3 ]; then
echo "starting testing results processing"
python process_game_experiment_from_file.py ../params/$id".params true"
echo "testing results processing complete"
fi

echo "finished"