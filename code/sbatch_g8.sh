#!/bin/bash

#SBATCH --job-name=g8_tp
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=400GB

## Example command to launch this script on JLab ifarm: \
##     sbatch -p gpu --nodes 2 --gres gpu:A100:4 sbatch_xxx.sh <py_script>

set -x

## Print info
env | grep -i slurm
env | grep -i rank
env | grep -i cuda
echo -e "=============================================================\n\n"

## HEAD node info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=32800
echo Head Node: $MASTER_ADDR:$MASTER_PORT
echo -e "=============================================================\n\n"

# Print CUDA device
srun --job-name print-cudevice --nodes 2 --ntasks-per-node 1\
	bash -c 'hostname; echo "CUDA_DEV: ${CUDA_VISIBLE_DEVICES}"'
echo -e "=============================================================\n\n"

## Torchrun
export PATH=$PATH:/work/epsci/xmei/projects/projects/pyvenv/bin  # add torchrun into PATH
export NCCL_DEBUG=INFO
env | grep -i nccl
# export NCCL_DEBUG_SUBSYS=NET

PY_SCRIPTNAME=$1

## Torchrun
### --nproc-per-node means GPUs per node
### For ifarm, use TCP ports 32768-60999
# --nproc_per_node=4 means 4 ranks per node. Each node has 4 GPUs.
srun torchrun --nproc_per_node=4 \
	--rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR.jlab.org:$MASTER_PORT \
  	--nnodes=2 \
	--rdzv-id $RANDOM \
	$1 1024 2

