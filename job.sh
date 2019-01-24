#!/bin/bash
#PBS -q h-debug              # <-------------- 投入するキューを指定する
#PBS -l select=2:mpiprocs=2:ompthreads=1  #<-- 実行したいノード数と、ノードあたりプロセス数を記述する（ここでは合計４プロセス）
#PBS -W group_list=pz0xxx    # <-------------- 自分のグループ名に書き換える
#PBS -l walltime=00:15:00    # <-------------- 実行時間上限を指定する

GROUP=$(id -ng)
MYDIR=/lustre/${GROUP}/${USER}
export HOME=$MYDIR

. $MYDIR/env.sh

cd $MYDIR

# デバッグ出力： Chainer と CuPy のバージョンを表示
python -c "import chainer; print(chainer.__version__)"
python -c "import cupy; print(cupy.cuda.nccl.get_version())"

mpiexec -x PYTHONUSERBASE \
        python ./train_mnist.py -g --communicator pure_nccl

