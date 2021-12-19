#!/bin/bash
#SBATCH --qos=batch
#SBATCH --mem=4GB
#SBATCH --time=01-00:00:00
#SBATCH --output=/fs/clip-realspeech/projects/vid_game/logs/batch_%j.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=craigtho@umiacs.umd.edu


type=$1
experiment=$2
num_runs=$3
gpu=$4
stage=$5

id=$SLURM_JOBID


. path.sh
export train_cmd="slurm.pl --config conf/slurm.conf"

source activate audneurorl

cd /fs/clip-realspeech/projects/vid_game/software/VidGameRL || exit
param_name=../params/$id".params"
echo "paramfile:"
echo $param_name

python create_params.py $param_name $type|| exit
echo "param file created"

if [[ "$type" == "cht" ]]
then


    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd --mem 16GB JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_cht.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "game" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_game.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "acousticgame" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_acousticgame.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "acousticgame_craig" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_acousticgame_craig.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "smartgame" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_smartgame.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "supervisedgame" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_supervisedgame.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$type" == "fullsupervision" ]]
then

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_fullsupervision.sh $stage $param_name || exit 1;
    wait
    echo "finished"
else
  echo "problem"
fi


