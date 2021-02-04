# RSMI，ZMの実験用リポジトリ
## 使用ライブラリ
- [Libtorch 1.4.0 CPU版](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip)←押したらダウンロードされます
- [Boost]( https://www.boost.org/)

## コンパイル
- 上記ライブラリのディレクトリをRSMIディレクトリと同じ階層に配置
    - もしくはMakefileのパスを変更
- `make -f Makefile`
    - 適宜 `make clean`

## データセット作り方
### 人工データ
- 例：1000000点の一様分布データセットを得たいとき

    `python data_generator.py -d uniform -s 1000000 -n 1 -f datasets/uniform_1000000_1_2_.csv -m 2`
- オプションの意味
    - -d : 分布
    - -s : サイズ（点の数）
    - -n : skewness(uniform,normalは1,skewedは4)
    - -f : ファイル名
    - -m : 次元数（2でしかやってません）
- ファイル名の付け方
    - "datasets/" + 分布 + "\_" + サイズ + "\_" + skeweness + "\_" + 次元数 + "_.csv"
### 自然データ
- [OpenStreetMap](https://download.geofabrik.de/)を使用するとき
    - リンクから，得たい地域のosmファイルをダウンロード（すごく時間かかるので注意）
    - osm2pgsqlを使用し，postgresqlにosmファイルをロード
        - planet_osm_point 等のテーブルが作られるはず
    - 人工データと同じ形でデータを取り出す
    - 例：planet_osm_pointテーブルから0~1座標の取り出し
        ~~~
        copy
        (
        select (ST_X(way)- min_x)/(max_x - min_x) as x,(ST_Y(way) - min_y)/(max_y - min_y)as y,ROW_NUMBER() OVER() - 1 from planet_osm_point a,
        (
        select max(ST_X(way)) as max_x , min(ST_X(way)) as min_x, max(ST_Y(way)) as max_y , min(ST_Y(way)) as min_y from planet_osm_point b
        )as sub
        )
        to 'D:\japan.csv' WITH CSV DELIMITER ',';
        ~~~
## 実行
- RSMIの例：

    `./Exp -c 1000000 -d uniform -s 1`
- ZMの例：

    `./Exp -c 1000000 -d uniform -s 1 -z`
- オプションの意味
    - -c : カーディナリティ（データサイズ）
    - -d : 分布
    - -s : skweness
    - -z : ZMで実行
## ↓オリジナル
# RSMI

##  How to use

### 1. Required libraries

#### LibTorch
homepage: https://pytorch.org/get-started/locally/

CPU version: https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip

For GPU version, you need to choose according to your setup.

#### boost

homepage: https://www.boost.org/

#### 2. Change Makefile

Choose CPU or GPU

```
# TYPE = CPU
TYPE = GPU

Change *home/liuguanli/Documents/libtorch_gpu* to your own path.

ifeq ($(TYPE), GPU)
	INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
	INCLUDE = -I/home/liuguanli/Documents/libtorch/include -I/home/liuguanli/Documents/libtorch/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch/lib
endif
```
#### 3. Change Exp.cpp

comment *#define use_gpu* to use CPU version

```C++
#ifndef use_gpu
#define use_gpu
.
.
.
#endif  // use_gpu
```

#### 4. Change path

Change the path is you do not want to store the datasets under the project's root path.

Constants.h
```C++
const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "./datasets/";
```

data_generator.py
```python
if __name__ == '__main__':
    distribution, size, skewness, filename, dim = parser(sys.argv[1:])
    if distribution == 'uniform':
        filename = "datasets/uniform_%d_1_%d_.csv"
        getUniformPoints(size, filename, dim)
    elif distribution == 'normal':
        filename = "datasets/normal_%d_1_%d_.csv"
        getNormalPoints(size, filename, dim)
    elif distribution == 'skewed':
        filename = "datasets/skewed_%d_%d_%d_.csv"
        getSkewedPoints(size, skewness, filename, dim)
```


#### 5. Prepare datasets

```bash
python data_generator.py -d uniform -s 1000000 -n 1 -f datasets/uniform_1000000_1_2_.csv -m 2
```

```bash
python data_generator.py -d normal -s 1000000 -n 1 -f datasets/normal_1000000_1_2_.csv -m 2
```

```bash
python data_generator.py -d skewed -s 1000000 -n 4 -f datasets/skewed_1000000_4_2_.csv -m 2
```

#### 6. Run

```bash
make clean
make -f Makefile
./Exp -c 1000000 -d uniform -s 1
./Exp -c 1000000 -d normal -s 1
./Exp -c 1000000 -d skewed -s 4
```

### Notions

model save. If you do not record the training time, you can use trained models and load them. 


```C++
//RSMI.h
    std::ifstream fin(this->model_path);
    if (!fin)
    {
	net->train_model(locations, labels);
	torch::save(net, this->model_path);
    }
    else
    {
	torch::load(net, this->model_path);
    }
```

### Paper

> Jianzhong Qi, Guanli Liu, Christian S. Jensen, Lars Kulik: [Effectively Learning Spatial Indices](http://www.vldb.org/pvldb/vol13/p2341-qi.pdf). Proc. VLDB Endow. 13(11): 2341-2354 (2020)

```tex
@article{DBLP:journals/pvldb/QiLJK20,
  author    = {Jianzhong Qi and
               Guanli Liu and
               Christian S. Jensen and
               Lars Kulik},
  title     = {Effectively Learning Spatial Indices},
  journal = {{PVLDB}}
  volume    = {13},
  number    = {11},
  pages     = {2341--2354},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2341-qi.pdf},
}
```

