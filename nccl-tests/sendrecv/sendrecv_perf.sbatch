#!/bin/bash

#SBATCH --job-name=2-GPU-bw
##SBATCH --ntasks-per-node=4           # Number of tasks (processes) per node
#SBATCH --mem=400G
#SBATCH --time=04:00:00               # Maximum 

# Command to run with: sbatch -p gpu --gres gpu:A100:4 --nodes 1 <curfile>.sbatch

set -x pipefail

# Load MPI module openmpi before launching this script
# module load mpi/openmpi-x86_64

# Slurm env 
env | grep SLURM

# NCCL config
export NCCL_HOME=${HOME}/projects/nccl/build
#export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
echo -e "\nNCCL env var"
env | grep NCCL


echo -e "\nMPI env var"
env | grep MPI

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NCCL_HOME}/lib:${MPI_LIB}

export WKDIR=${HOME}/projects/nccl-tests

#### Only 1 process launched
# Task 1: GPU 0-1 BW
CUDA_VISIBLE_DEVICES=0,1 ${WKDIR}/build/sendrecv_perf -b 8 -e 1G -f 1.2 -g 2   # Run on 1 node only

# Task 2: GPU 0-2 BW
CUDA_VISIBLE_DEVICES=0,2 ${WKDIR}/build/sendrecv_perf -b 8 -e 1G -f 1.2 -g 2   # Run on 1 node only
