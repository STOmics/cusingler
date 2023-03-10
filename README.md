# cuSingleR

基于python版本SingleR软件,使用cuda加速.

开发中...

## 依赖库

* gcc 11.3
* cmake 3.25.2
* hdf5 1.12.1

推荐使用conda安装依赖库

```sh
conda install -c conda-forge hdf5 cmake gcc cxx-compiler
```

## 编译

进入代码仓库根目录,创建build目录,在其中进行编译,编译后的可执行和依赖库放在install目录下.

需要设置gcc/hdf5等库路径.

```sh
mkdir build && cd build

export HDF5_PATH=/home/fxzhao/anaconda3/envs/singlerr
export CPLUS_INCLUDE_PATH=$HDF5_PATH/include:/home/fxzhao/anaconda3/envs/py/include
export LD_LIBRARY_PATH=$HDF5_PATH/lib
export LIBRARY_PATH=$LD_LIBRARY_PATH

cmake .. --fresh -DCMAKE_INSTALL_PREFIX=../install -DHDF5_LIB=${HDF5_PATH}/lib
make
```