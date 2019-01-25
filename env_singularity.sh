#!/bin/bash

GROUP=$(id -ng)
MYDIR=/lustre/${GROUP}/$(id -nu)

unset PIP_ROOT
export PYTHONUSERBASE=$MYDIR/.python
export LC_ALL="C"
export SINGULARITY_CACHEDIR=$MYDIR/.singularity

export HOME=$MYDIR

export PATH=/lustre/app/singularity/2.5.1/bin:$PATH

MPI_ROOT=/lustre/app/openmpi/2.1.2/ofed4.2/gnu
CUDA_ROOT=/lustre/app/acc/cuda/9.1.85

export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/lib:/lib/x86_64-linux-gnu:/usr/lib:/usr/lib64:$CUDA_ROOT/lib64:/usr/local/nvidia-396.44:/opt/mellanox/sharp/lib:/opt/mellanox/mxm/lib:$LD_LIBRARY_PATH



