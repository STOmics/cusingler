# cuSingleR

基于python版本SingleR软件,使用cuda加速.

## 编译whl

系统环境需要安装 cuda-toolkit 11.4 / gcc<11

```sh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python setup.py sdist bdist_wheel
```

编译后的whl存放在dist目录

## 运行

```sh
python test/test_basic.py data/GSE84133_GSM2230761_mouse1.h5ad data/GSE84133_GSM2230762_mouse2.h5ad data/result.tsv
```

参数解释:

* 第一个参数表示reference的数据,格式为h5ad
* 第二个参数表示query的数据,格式为h5ad
* 第三个参数为输出文件,按tab分割的三列数据,分别表示cell/firstLabel/finalLabel

详情见脚本 *test/test_basic.py*

注意: 由于h5ad格式可能存在不同字段,如果脚本报错需要修改正确的字段以保证程序能获取到数据

## 输入文件要求

* ref/qry输入文件均需有/X/data /X/indices /X/indptr 存储矩阵数据; 有/var/_index 存储基因名
* ref需有 /obs/ClusterName/codes /obs/ClusterName/categories 或/obs/celltype/codes /obs/celltype/categories 存储分类结果
* qry需有 /obs/_index 存储细胞名

* 程序不会对h5ad数据做log2处理,如有此需要请客户自行对数据做预处理
* 如遇ref和qry的gene不完全一致,程序会自动取基因的交集参与计算,如果没有一个相同基因,则程序报错退出