#!/bin/bash

#SBATCH --job-name=2-node_gpt
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000   # MB

set -euxo pipefail

env | grep -i slurm
echo "============================================================="

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo $nodes
echo
echo HEAD Node IP: $head_node_ip

export PATH=$PATH:/work/epsci/xmei/projects/yifan_sun/py-torch/bin  # add torchrun into PATH
export LOGLEVEL=INFO

srun torchrun \
	--nnodes 2 \
	--nproc-per-node 2 \
	--rdzv-id $RANDOM \
	--rdzv-backend c10d \
	--rdzv-endpoint $head_node_ip:60010 \
	transformer_ddp.py 2
