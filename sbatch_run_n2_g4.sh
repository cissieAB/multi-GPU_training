#!/bin/bash

#SBATCH --job-name=2-node_gpt
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000   # MB

##  Command to launch this script on JLab ifarm: salloc -p gpu 

set -euxo pipefail

## Print Slurm info
env | grep -i slurm
echo "============================================================="

## Get head node IP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo $nodes
echo
echo HEAD Node IP: $head_node_ip

## Torchrun
export PATH=$PATH:/work/epsci/xmei/projects/yifan_sun/py-torch/bin  # add torchrun into PATH
export LOGLEVEL=INFO

## --nproc-per-node means GPUs per node
## For ifarm, use TCP ports 32768-60999
srun torchrun \
	--nnodes 2 \
	--nproc-per-node 4 \
	--rdzv-id $RANDOM \
	--rdzv-backend c10d \
	--rdzv-endpoint $head_node_ip:60010 \
	transformer_ddp.py 2
