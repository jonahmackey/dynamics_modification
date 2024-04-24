#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --mail-user=jonah.mackey@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:2
#SBATCH --tasks-per-node=2
#SBATCH --mem=16G
#SBATCH --time=0-03:00

MODEL_NAME=$1
DATASET_NAME=$2
LR=$3
EXPERIMENT_TYPE=$4
RESULTS_PATH=$5

mkdir "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_lr=${LR}_${SLURM_JOB_ID}"
module load python/3.10
module load scipy-stack
source /home/jmackey/dm/bin/activate

export NCCL_BLOCKING_WAIT=1
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

python /home/jmackey/scratch/dynamics_modification/experiments/distributed_finetuning.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --data_path /home/jmackey/datasets \
    --lr ${LR} \
    --batch_size 128 \
    --num_iters 1000 \
    --warmup_steps 200 \
    --print_every 100 \
    --exp_type ${EXPERIMENT_TYPE} \
    --heads_path /home/jmackey/scratch/dynamics_modification/heads \
    --results_path "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_lr=${LR}_${SLURM_JOB_ID}" \
    --job_id "${SLURM_JOB_ID}" \
    --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))