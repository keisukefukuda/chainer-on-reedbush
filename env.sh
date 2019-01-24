#!/bin/bash

GROUP=$(id -ng)
MYDIR=/lustre/${GROUP}/$USER

. /etc/profile.d/modules.sh

module load cuda9/9.1.85
module load anaconda3/4.3.0
module load openmpi/gdr/2.1.2/gnu

unset PIP_ROOT
export PYTHONUSERBASE=$MYDIR/.python

