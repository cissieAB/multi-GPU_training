#!/bin/bash

#SBATCH --job-name=1-node_gpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000   # MB

## Example command to launch this script on JLab ifarm: sbatch -p gpu --gres gpu:A100:4 sbatch_run_xxx.sh 

set -x

## Print info
env | grep -i slurm
env | grep -i rank
env | grep -i cuda
env | grep -i nccl
echo -e "=============================================================\n\n"

## Host info
srun --job-name hostname --nodes 1 --ntasks-per-node 1 hostname

## HEAD node info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=32800
echo Head Node: $MASTER_ADDR:$MASTER_PORT
echo -e "=============================================================\n\n"

## Torchrun
export PATH=$PATH:/work/epsci/xmei/projects/yifan_sun/py-torch/bin  # add torchrun into PATH
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3

## NCCL
export PYTHON_DIST_JOB_ARGS="-m torch.distributed.run --nproc_per_node=$SLURM_GPUS_ON_NODE --nnodes=$SLURM_NNODES --master-addr $head_node_ip --master-port $PORT_NUM"
# srun --job-name nccl-test python $PYTHON_DIST_JOB_ARGS nccl_test.py

srun --job-name print-cuda --nodes 1 --ntasks-per-node 1 echo $(hostname) ${CUDA_VISIBLE_DEVICES}
echo -e "=============================================================\n\n"


## Torchrun
### --nproc-per-node means GPUs per node
### For ifarm, use TCP ports 32768-60999
# --nproc_per_node=4 means 4 ranks per node. Each node has 4 GPUs.
srun torchrun --nproc_per_node=4 \
	--rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR.jlab.org:$MASTER_PORT \
  	--nnodes=1 \
	--rdzv-id $RANDOM \
	transformer_ddp.py 2

