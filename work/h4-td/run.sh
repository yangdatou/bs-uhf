#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=BS-NOCI
#SBATCH --mem=0

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

export OMP_NUM_THREADS=28;
export MKL_NUM_THREADS=28;
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export TMPDIR=/scratch/global/yangjunjie/
export PYSCF_TMPDIR=/scratch/global/yangjunjie/

export PYSCF_TMPDIR=/scratch/global/yangjunjie/;
export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/;
export PYTHONPATH=/home/yangjunjie/work/bs-uhf/src/:$PYTHONPATH;
export PYTHONPATH=/home/yangjunjie/packages/wick/wick-dev/:$PYTHONPATH;

export PYTHONUNBUFFERED=TRUE;

python h4-td.py

