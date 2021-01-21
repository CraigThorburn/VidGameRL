#!/bin/bash
#SBATCH --qos=shallow
#SBATCH --mem=4GB
#SBATCH --time=02-00:00:00
#SBATCH --output=/fs/clip-realspeech/projects/vid_game/logs/batch_%j.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=craigtho@umiacs.umd.edu



experiment=$1
num_runs=$2
slurm=$3
gpu=$4
params=$5
echo $params
. path.sh
export train_cmd="slurm.pl --config conf/slurm.conf"


source activate audneurorl

if [[ "$slurm" == "true" ]]
then
    echo "comencing slurm batch parallel across "$num_runs" machines"

    $train_cmd --mem 16GB --time 01-00:00:00 JOB=1:$num_runs --gpu $gpu ../../data/$experiment/log/main_game.$SLURM_JOBID.JOB.log  run_batch_individual.sh $params || exit 1; 
    wait
    echo "finished"

elif [[ "$slurm" == "false" ]]
then
        echo "comencing batch without slurm"
	for i in $(eval echo "{1..$num_runs}")
        do
                echo "---------------------"
                echo "launching run "$i
		module add cuda
		python create_param_file.py ../params/$id".params" $SLURM_JOBID
		python main_game_LSTM.py ../params/$id".params"
                python main_game.py $host -overwrite $overwrite -modelname run$i
        echo "finished"
        done
else
  echo "problem"
fi


