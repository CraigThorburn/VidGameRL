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

#if [ $stage -le 0 ]; then
#echo "starting pretraining"
#python acousticgame_pretrain_network.py $params -run_num=$i || exit
#echo "pretraining complete"
#fi

#if [ $stage -le 1 ]; then
#echo "starting pretrain validation"
#python acousticgame_pretrain_validation.py $params -run_num=$i || exit
#echo "pretrain validation complete"
#fi

#if [ $stage -le 2 ]; then
#echo "calculating fischer coefficients"
#python acousticgame_calculate_ewc_coeffs.py $params -run_num=$i || exit
#fi


if [ $stage -le 3 ]; then
echo "starting training from pretrained model"
python acousticgame_train.py $params -run_num=$i || exit
echo "training complete"
fi

if [ $stage -le 4 ]; then
echo "starting training results processing"
python game_process_experiment.py $params "false" -run_num=$i || exit
echo "training results processing complete"
fi


#if [ $stage -le 5 ]; then
#echo "starting abx for both of last two layers"
#python acousticgame_run_abx.py $params -run_num=$i -layer=-1 -pretrain=false || exit
#python acousticgame_run_abx.py $params -run_num=$i -layer=-2 -pretrain=false || exit
#python acousticgame_run_abx.py $params -run_num=$i -layer=-3 -pretrain=false || exit
#python acousticgame_run_abx.py $params -run_num=$i -layer=-4 -pretrain=false || exit
#echo "abx complete"
#fi

#if [ $stage -le 6 ]; then
#echo "starting testing from pretrained model"
#python acousticgame_test.py $params -run_num=$i
#echo "testing complete"
#fi

#if [ $stage -le 7 ]; then
#echo "starting testing results processing"
#python game_process_experiment.py $params "true" -run_num=$i
#echo "testing results processing complete"
#fi

echo "finished"