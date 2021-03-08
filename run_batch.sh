#!/bin/bash
#SBATCH --qos=batch
#SBATCH --mem=4GB
#SBATCH --time=01-00:00:00
#SBATCH --output=/fs/clip-realspeech/projects/vid_game/logs/batch_%j.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=craigtho@umiacs.umd.edu


pretrain=$1
experiment=$2
num_runs=$3
gpu=$4
stage=$5

id=$SLURM_JOBID


. path.sh
export train_cmd="slurm.pl --config conf/slurm.conf"

source activate audneurorl

if [[ "$pretrain" == "false" ]]
then
    cd /fs/clip-realspeech/projects/vid_game/software/VidGameRL || exit
    param_name=../params/$id".params"
    echo "paramfile:"
    echo $param_name

    python create_params_file.py $param_name || exit
    echo "param file created"

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd --mem 16GB --time 01-00:00:00 JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_nopretrain.sh $stage $param_name || exit 1;
    wait
    echo "finished"

elif [[ "$pretrain" == "true" ]]
then
    cd /fs/clip-realspeech/projects/vid_game/software/VidGameRL || exit
    param_name=../params/$id".params"
    echo "paramfile:"
    echo param_name

    python create_params_file.py $param_name || exit
    echo "param file created"

    echo "comencing slurm batch parallel across " $num_runs " machines"

    $train_cmd --mem 16GB --time 01-00:00:00 JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual_pretrain.sh $stage $param_name || exit 1;
    wait
    echo "finished"

else
  echo "problem"
fi


