#!/bin/bash
#SBATCH --qos=batch
#SBATCH --mem=4GB
#SBATCH --time=01-00:00:00
#SBATCH --output=/fs/clip-realspeech/projects/vid_game/logs/batch_%j.txt
#SBATCH --mail-type=fail
#SBATCH --mail-user=craigtho@umiacs.umd.edu

input="$@"
experiment_name=$1
data_folder=$2
params=$3
model_id=$4
gpu=$5
stage=$6
num_runs=$7

. path.sh
export train_cmd="slurm.pl --config conf/slurm.conf"

echo "experiment_name:"
echo $experiment_name
echo 'data_folder: '
echo $data_folder
echo 'params: '
echo $params
echo "model_id:"
echo $model_id
echo 'gpu: '
echo $gpu
echo 'stage: '
echo $stage
echo 'num_runs: '
echo $num_runs
source activate audneurorl_test

echo "---------------------"

if [ $stage -le 0 ]; then
echo "starting pretraining"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_pretrain_network.py $params || exit 1;
   echo "pretraining complete"
fi

if [ $stage -le 1 ]; then
echo "starting pretraining validation"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_validation.$model_id.JOB.log  run_python.sh run_validation.py "$params -pretrain=true" || exit 1;
   echo "pretraining complete validation"
fi

if [ $stage -le 2 ]; then
echo "starting calculating ewc coefficients"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_calculate_ewc_coeffs.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py $params || exit 1;
   echo "starting calculating ewc coefficients"
fi

if [ $stage -le 3 ]; then
echo "running pretrain abx task"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_run_abxpretrain-1.$model_id.JOB.log  run_python.sh run_abx.py "$params -layer=-1 -pretrain=true" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-2 -pretrain=true" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-3 -pretrain=true" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-4 -pretrain=true" || exit 1;
echo "abx complete"
fi


if [ $gpu -eq 1 ]; then
train_gpu=3
fi

if [ $gpu -eq 2 ]; then
train_gpu=4
fi


if [ $stage -le 4 ]; then
echo "starting training from pretrained model"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $train_gpu ../../data/$data_folder/log/$experiment_name/acousticgame_train.$model_id.JOB.log  run_python.sh acousticgame_train.py $params || exit 1;
   echo "training complete"
fi



if [ $stage -le 5 ]; then
echo "starting training results processing"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/game_process_experiment.$model_id.JOB.log  run_python.sh game_process_experiment.py "$params false" || exit 1;
   echo "training results processing complete"
fi



if [ $stage -le 6 ]; then
echo "running training abx task"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_run_abxtrain-1.$model_id.JOB.log  run_python.sh run_abx.py "$params -layer=-1 -pretrain=false" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-2 -pretrain=false" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-3 -pretrain=false" || exit 1;
#../../data/$data_folder/log/$experiment_name/acousticgame_pretrain_network.$model_id.JOB.log  run_python.sh acousticgame_calculate_ewc_coeffs.py "$params -layer=-4 -pretrain=false" || exit 1;
echo "reaining abx task complete"
fi

#if [ $stage -le 7 ]; then
#echo "starting testing from pretrained model"
#$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_test.$model_id.JOB.log  #run_python.sh acousticgame_test.py $params || exit 1;
#   echo "testing complete"
#fi

#if [ $stage -le 8 ]; then
#echo "starting testing results processing"
#$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/game_process_experiment.$model_id.JOB.log  #run_python.sh game_process_experiment.py "$params true" || exit 1;
#   echo "testing results processing complete"
#fi


if [ $stage -le 9 ]; then
echo "starting training validation"
$train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$data_folder/log/$experiment_name/acousticgame_train_validation.$model_id.JOB.log  run_python.sh run_validation.py "$params -pretrain=true" || exit 1;
   echo "training complete validation"
fi

echo "finished"