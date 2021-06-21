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
python cht_train.py $params -run_num=$i
echo "training complete"
fi

echo "finished"