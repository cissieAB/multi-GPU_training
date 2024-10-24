#!/bin/bash

#SBATCH --job-name=nccl-test
#SBATCH --ntasks-per-node=4           # Number of tasks (processes) per node
#SBATCH --mem=400G
#SBATCH --time=00:30:00               # Maximum 

# Command to run with: sbatch -p gpuu --gres gpu:A100:4 --nodes <n> <curfile>.sbatch

set -x pipefail

# Load MPI module openmpi
module load mpi/openmpi-x86_64

# Slurm env 
env | grep SLURM

# NCCL config
export NCCL_HOME=${HOME}/projects/nccl/build
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
echo -e "\nNCCL env var"
env | grep NCCL


echo -e "\nMPI env var"
env | grep MPI

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NCCL_HOME}/lib:${MPI_LIB}

export WKDIR=${HOME}/projects/nccl-tests

mpirun -np ${SLURM_NPROCS} ${WKDIR}/build/all_reduce_perf -b 128M -e 2G -f 2 -g 1
# ${WKDIR}/build/all_reduce_perf -b 128M -e 8G -f 2 -g 4   # Run on 1 node only

