#!/bin/bash

EXPERIMENT_TITLE=$1
EXPERIMENT_TYPE=$2

echo "Creating experiment directory /home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"
mkdir "/home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"

MODEL_NAMES=( "ViT-B-32" "ViT-B-16" )
DATASET_NAMES=( "MNIST" )
LR_VALS=( 0.01 )

for MODEL_NAME in "${MODEL_NAMES[@]}";
do
    for DATASET_NAME in "${DATASET_NAMES[@]}";
    do
        for LR in "${LR_VALS[@]}";
        do
            SBATCH_COMMAND="/home/jmackey/scratch/dynamics_modification/scripts/single_job.sh \
                ${MODEL_NAME} \
                ${DATASET_NAME} \
                ${LR} \
                ${EXPERIMENT_TYPE} \
                /home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"
            echo "Running sbatch command ${SBATCH_COMMAND}"
            sbatch ${SBATCH_COMMAND}
        done
    done
done

echo "${EXPERIMENT_TYPE} experiments submitted!"
            

