# 参考資料
 * Reedbush Quick Start Guide https://www.cc.u-tokyo.ac.jp/supercomputer/files/QuickStartGuide.pdf
 * 利用ポータル https://reedbush-www.cc.u-tokyo.ac.jp/
 
# 手順（１）
Quick Start Guideの項目4〜14を参考に、初期設定を行い、ログインできることを確認する

# 手順（２）
以下の手順で、 CUDA 9.1 上に Chainer 5.1.0をインストールできる。 現時点では、CUDA 9.2 のmoduleで一部問題があったので9.1を利用している。

## 環境設定
計算ノードからはHOMEは見えないので、Lustreを事実上のHOMEとして用いる。
```
$ cd /lustre/$(id -ng)/$USER/
```

環境設定ファイル `env.sh` を作成する（内容は、このgistの別ファイル参照）

```
$ vi env.sh
```

env.shを読み込む。この操作はログインのたびに毎回行うので、 `.bash_profile` 等に書くと良い。

```
$ . /lustre/$(id -ng)/$USER/env.sh
```

## pip
次に、pipで必要なモジュールをインストールする。env.sh内に定義されている `PYTHONUSERBASE` の定義により、 Lustre下にインストールされる

```
# pip install は、ファイルシステムの調子によって、数分〜5分かかる場合があるので気長に待つ
$ pip install --user mpi4py
$ pip install --user cupy-cuda91==5.1.0
$ pip install --user chainer==5.1.0
```

## MNISTの実行準備

ChainerのMNISTサンプルは、初回実行時に`HOME`ディレクトリにMNISTデータをダウンロードする。計算ノードからはインターネットにアクセスできないので、ログインノードでMNISTを一回実行してデータをダウンロードさせる。この実行はChainerMNである必要はない。（まちがえてmasterブランチのtrain_mnist.pyをダウンロードするとエラーで実行できないので注意）
（なお、手動でデータをコピーしても良い）

```
# NOTE: env.shの実行を忘れないように
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/examples/mnist/train_mnist.py -O train_mnist_single.py
$ python train_mnist_single.py -e 1
```

次に、ChainerMN用の train_mnist.py をダウンロードする。（まちがえてmasterブランチのtrain_mnist.pyをダウンロードするとエラーで実行できないので注意）

```
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/examples/chainermn/mnist/train_mnist.py
```

最後に、ジョブを実行するためのジョブスクリプトを記述する（内容はこのGistの別ファイル参照）

```
$ vi job.sh
```

## MNISTの実行と確認

ジョブを投入する。

```
$ chmod +x job.sh
$ qsub job.sh
```

ジョブが投入されたことを確認する

```
$ rbstat 
Reedbush-H scheduled stop time: 2019/01/29(Tue) 09:00:00 (Remain: 18days 17:20:19)
Reedbush-L scheduled stop time: 2019/01/29(Tue) 09:00:00 (Remain: 18days 17:20:19)
Reedbush-U scheduled stop time: 2019/01/29(Tue) 09:00:00 (Remain: 18days 17:20:19)

JOB_ID     JOB_NAME   STATUS  PROJECT    QUEUE           START_DATE       ELAPSE   TOKEN    NODE
1405000    job.sh     RUNNING pz0425     h-debug         01/10 15:39:08   00:00:24 0.1      2
```

暫く待つとジョブが終了するので、結果を確認する

```
$ cat job.sh.o1405000
5.1.0
2115
[1547102403.645058] [a086:18043:0]         sys.c:744  MXM  WARN  Conflicting CPU frequencies detected, using: 2600.72
[1547102403.645032] [a086:18044:0]         sys.c:744  MXM  WARN  Conflicting CPU frequencies detected, using: 2600.72
[1547102405.171371] [a087:22918:0]         sys.c:744  MXM  WARN  Conflicting CPU frequencies detected, using: 2602.44
[1547102405.171374] [a087:22919:0]         sys.c:744  MXM  WARN  Conflicting CPU frequencies detected, using: 2602.44
==========================================
Num process (COMM_WORLD): 4
Using GPUs
Using pure_nccl communicator
Num unit: 1000
Num Minibatch-size: 100
Num epoch: 20
==========================================
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.296747    0.123968              0.912467       0.9644                    5.49285
2           0.0975135   0.0798887             0.971667       0.9751                    6.85029
3           0.0583089   0.0665895             0.982667       0.9792                    8.19364
4           0.0378759   0.066392              0.988534       0.9797                    9.50622
5           0.028316    0.063212              0.991          0.9808                    10.8018
6           0.0177264   0.070744              0.9954         0.9804                    12.0978
7           0.0113274   0.0690856             0.996733       0.981                     13.3992
8           0.0106449   0.0800302             0.996067       0.9783                    14.7028
9           0.0119948   0.0708811             0.996267       0.9802                    16.0077
10          0.0135936   0.0718536             0.995533       0.9824                    17.3134
11          0.0102226   0.0728081             0.996334       0.9824                    18.6169
12          0.00913799  0.0785161             0.996867       0.9815                    19.9324
13          0.00474206  0.0799097             0.998267       0.9819                    21.279
14          0.00706878  0.0834321             0.997867       0.9822                    22.6339
15          0.00450557  0.0818857             0.998533       0.9838                    23.9803
16          0.00815668  0.0878528             0.996733       0.9812                    25.3374
17          0.0154402   0.079348              0.9958         0.9822                    26.6402
18          0.00871023  0.0872598             0.996667       0.9789                    27.9572
19          0.00753859  0.0731627             0.997467       0.9842                    29.278
20          0.00245414  0.0770232             0.9994         0.9852                    30.596


```



# Singularityの利用

ログインノード上でイメージをビルドする

```
$ . env.sh
$ cd $HOME
$ singularity build chainer.img docker://keisukef/chainer-reedbush:cuda91-chainer5.1.0
```

GPUを利用するため、インタラクティブジョブを投げてみます

```
$ qsub -I -q h-interactive -l select=1:mpiprocs=2 -W group_list=$(id -ng) -l walltime=0:30:00
qsub: waiting for job 1425591.reedbush-pbsadmin0 to start
qsub: job 1425591.reedbush-pbsadmin0 ready

[z30425@a090 ~]$ cd /lustre/$(id -ng)/$USER/
[z30425@a090 ~]$ . env.sh
[z30425@a090 ~]$ singularity exec --nv chainer.img /lustre/app/openmpi/2.1.2/ofed4.2/gnu/bin/mpiexec -n 2 bash -c ". env_singularity.sh; /usr/local/python3.6/bin/python3 train_mnist.py -g"
```


