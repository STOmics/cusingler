# cuSingleR

基于python版本SingleR软件,使用cuda加速.

开发中...

## 依赖库

* gcc/g++ 9.4.0 
* cmake 3.25.2
* hdf5 1.12.1
* cuda-toolkit 11.4

推荐使用conda安装依赖库

```sh
conda create -n cs python=3.8
conda activate cs
conda install -c conda-forge mamba
mamba install -c conda-forge hdf5 cmake gcc==9.4.0 gxx==9.4.0 gxx_linux-64=9.4 gcc_linux-64=9.4 sysroot_linux-64=2.17
```

## 编译

进入代码仓库根目录,创建build目录,在其中进行编译,编译后的可执行和依赖库放在install目录下.

需要设置gcc/hdf5等库路径,将上述conda环境替换为环境变量 $condaPath

```sh
mkdir build && cd build

condaPath=/home/fxzhao/anaconda3/envs/singlerr

export PATH=/usr/local/cuda/bin:$condaPath/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin

export HDF5_PATH=$condaPath
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/:$HDF5_PATH/include

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HDF5_PATH/lib
export LIBRARY_PATH=$LD_LIBRARY_PATH

cmake .. --fresh -DCMAKE_INSTALL_PREFIX=../install -DHDF5_LIB=${HDF5_PATH}/lib
make
```

## 运行

```sh
./install/bin/cusingler ./data/small_input.h5
```