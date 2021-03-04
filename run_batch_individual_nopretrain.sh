#SBATCH --qos=dque
#SBATCH --mem-per-cpu=16GB 
#SBATCH --time=01-00:00:00

i=$SLURM_ARRAY_TASK_ID
input="$@"
stage=$1
params=$2
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

if [ $stage -le 0 ]; then
echo "starting training"
python main_game_conv.py $params -run_num=$i
echo "training complete"
fi

if [ $stage -le 1 ]; then
echo "starting training results processing"
python process_game_experiment.py $params "false" -run_num=$i
echo "training results processing complete"
fi

if [ $stage -le 2 ]; then
echo "starting testing"
python test_conv.py $params -run_num=$i
echo "testing complete"
fi

if [ $stage -le 3 ]; then
echo "starting testing results processing"
python process_game_experiment.py $params "true" -run_num=$i
echo "testing results processing complete"
fi

echo "finished"