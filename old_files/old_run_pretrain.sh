#SBATCH --qos=dque
#SBATCH --mem-per-cpu=16GB 

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
echo "starting pretraining"
python pretrain_network.py $params -run_num=$i
echo "pretraining complete"
fi

if [ $stage -le 1 ]; then
echo "starting pretrain validation"
python pretrain_validation.py $params -run_num=$i
echo "pretrain validation complete"
fi

if [ $stage -le 2 ]; then
echo "starting training from pretrained model"
python main_game_from_pretrain.py $params -run_num=$i
echo "training complete"
fi

if [ $stage -le 3 ]; then
echo "starting training results processing"
python process_game_experiment.py $params "false" -run_num=$i
echo "training results processing complete"
fi


if [ $stage -le 4 ]; then
echo "starting abx for both of last two layers"
python run_abx.py $params -run_num=$i -layer=-1 -pretrain=false
python run_abx.py $params -run_num=$i -layer=-2 -pretrain=false
python run_abx.py $params -run_num=$i -layer=-3 -pretrain=false
python run_abx.py $params -run_num=$i -layer=-4 -pretrain=false
echo "abx complete"
fi

if [ $stage -le 5 ]; then
echo "starting testing from pretrained model"
python test_conv_from_pretrain.py $params -run_num=$i
echo "testing complete"
fi

if [ $stage -le 6 ]; then
echo "starting testing results processing"
python process_game_experiment.py $params "true" -run_num=$i
echo "testing results processing complete"
fi

echo "finished"