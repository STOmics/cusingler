# cuSingleR

基于python版本SingleR软件,使用cuda加速.

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
mamba install -c conda-forge cli11
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
./install/bin/cusingler \
    -r ./data/5w_ref.h5ad \
    -q ./data/Mouse_brain_D1_qry.h5ad \
    -o stat.tsv \
    -c 20 \
    -g 0
```

参数解释:

* 第一个参数表示reference的数据,格式为h5ad
* 第二个参数表示query的数据,格式为h5ad
* 第三个参数为输出文件,按tab分割的三列数据,分别表示cell/firstLabel/finalLabel
* 第四个参数20表示使用的cpu核数,默认为自动检测
* 第五个参数0表示使用第0块GPU卡,默认为0,可设置0,1,2等实际存在的GPU卡

## 输入文件要求

* ref/qry输入文件均需有/X/data /X/indices /X/indptr 存储矩阵数据; 有/var/_index 存储基因名
* ref需有 /obs/ClusterName/codes /obs/ClusterName/categories 或/obs/celltype/codes /obs/celltype/categories 存储分类结果
* qry需有 /obs/_index 存储细胞名

* 程序不会对h5ad数据做log2处理,如有此需要请客户自行对数据做预处理
* 如遇ref和qry的gene不完全一致,程序会自动取基因的交集参与计算,如果没有一个相同基因,则程序报错退出