# 语音合成学习（十一）学习笔记

---
# Milvus的以文搜图

## 一、介绍

文搜图指的是，根据文本描述，从图像数据库中检索与文本内容相似的图像数据并返回。

### Chinese-CLIP
>  CLIP(Contrastive Language Image Pretraining)是[OpenAI](https://so.csdn.net/so/search?q=OpenAI&spm=1001.2101.3001.7020)在2021年提出的多模态神经网络模型，它是一种可以同时处理文本和图像的预训练模型。与以往的图像分类模型不同，Clip并没有使用大规模的标注图像数据集来进行训练，而是通过自监督学习的方式从未标注的图像和文本数据中进行预训练，使得模型能够理解图像和文本之间的语义联系。
>
>  该模型基于OpenAI 收集到的 4 亿对图像文本对进行训练，分别将文本和图像进行编码，之后使用 metric learning 进行权重，其目标是将图像与文本的相似性提高，大致如下图所示。具体内容不在本文赘述。
>
> 
>  <img align=“center” src="/img/clip.png"/>
>
>  这个模型包含3步：
>
>  1）文本和图片两路分别进行特征编码；
>
>  2）将文本、图片的特征从各自的单模态特征空间投射到一个多模态的特征空间；
>
>  3）在多模态的特征空间中，原本成对的图像文本（正样本）的特征向量之间的距离应该越近越好，互相不成对的图像文本（负样本）的特征向量之间的距离应该越远越好。
>
>  
>
>   Chinese-CLIP是CLIP模型的中文版本，使用大规模的图文对进行训练得到，针对中文领域数据能够实现更好的效果。本文的以文搜图应用使用中文CLIP作为文本和图像的特征提取器，提取数据集中所有图像的特征作为数据库，用于检索。

------

## 快速使用

**本地部署**

```bash
# 通过pip安装
pip install cn_clip

# 或者从源代码安装
cd Chinese-CLIP
pip install -e .

# 随后就是几行代码就可以快速产出特征，做个简单的零样本分类只需要把预先准备好的标签传入即可：

import torch

from PIL import Image

import cn_clip.clip as clip

from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models()) 

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

model.eval()

image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)

text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():

    image_features = model.encode_image(image)

    text_features = model.encode_text(text)

    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务

    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_features /= text_features.norm(dim=-1, keepdim=True)   

    logits_per_image, logits_per_text = model.get_similarity(image, text)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]

# 文件夹会生成一个pokemon.jpeg图片文件
encode_image和encode_text分别负责产出图像和文本特征，随后完成归一化以后，用get_similarity的接口就可以算出图像文本的相似度，从而得出图像的分类概率。
```

---

## 使用Python运行Milvus

- **服务器部署**Milvus服务（Install Milvus Standalone with Docker Compose）

> **1. [下载](https://link.zhihu.com/?target=https%3A//github.com/milvus-io/milvus/releases/download/v2.2.4/milvus-standalone-docker-compose.yml)`milvus-standalone-docker-compose.yml`保存为`docker-compose.yml`**
>
> ```bash
> # Download the YAML file
> # Download milvus-standalone-docker-compose.yml and save it as docker-compose.yml manually, or with the following command.
> wget https://github.com/milvus-io/milvus/releases/download/v2.2.8/milvus-standalone-docker-compose.yml -O docker-compose.yml
> ```
>
> **2.Start Milvus**
>
> ```bash
> # 在与 docker-compose. yml 文件相同的目录中，运行以下命令启动 Milvus:
> sudo docker-compose up -d
> 
> # 如果您的系统安装了 Docker Compose V2而不是 V1，那么使用 Docker Compose 而不是 Docker-Comose。检查 $docker 撰写版本是否是这种情况。
> ```
>
> **3.检查容器是否已经启动并运行。**
>
> ```bash
> sudo docker-compose ps
> # Milvus 独立启动后，将运行三个 Docker 容器，包括 Milvus 独立服务及其两个依赖项。
> 
> 
> # 停止Milvus 
> sudo docker-compose down
> # 停止Milvus后删除数据，使用命令:
> sudo rm -rf  volumes
> ```
>
> <img align=“center” src="/img/image-docker-milvus.png"/>

# Milvus

​	Milvus是一个向量数据库，基于深度学习网络提取的特征进行对象之间相似度计算，返回topK个相似的结果。

## 基本概念

### **特征向量**

向量又称为 embedding vector，是指由 embedding 技术从离散变量（如图片、视频、音频、自然语言等各种非结构化数据）转变而来的连续向量。

在数学表示上，向量是一个由浮点数或者二值型数据组成的 n 维数组。

通过现代的向量转化技术，比如各种人工智能（AI）或者机器学习（ML）模型，可以将非结构化数据抽象为 n 维特征向量空间的向量。这样就可以采用最近邻算法（ANN）计算非结构化数据之间的相似度。

### **向量相似度检索**

相似度检索是指将目标对象与数据库中数据进行比对，并召回最相似的结果。同理，向量相似度检索返回的是最相似的向量数据。

近似最近邻搜索（ANN）算法能够计算向量之间的距离，从而提升向量相似度检索的速度。如果两条向量十分相似，这就意味着他们所代表的源数据也十分相似。

### **Collection-集合**

包含一组 entity，可以等价于关系型数据库系统（RDBMS）中的表。

### **Entity-实体**

包含一组 field。field 与实际对象相对应。field 可以是代表对象属性的结构化数据，也可以是代表对象特征的向量。primary key 是用于指代一个 entity 的唯一值。

**注意：** 你可以自定义 primary key，否则 Milvus 将会自动生成 primary key。请注意，目前 Milvus 不支持 primary key 去重，因此有可能在一个 collection 内出现 primary key 相同的 entity。

### **索引**

索引基于原始数据构建，可以提高对 collection 数据搜索的速度。Milvus 支持多种**[索引类型](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/index.md)**。为提高查询性能，你可以为每个向量字段指定一种索引类型。目前，一个向量字段仅支持一种索引类型。切换索引类型时，Milvus 自动删除之前的索引。

**相似性搜索引擎的工作原理**是将输入的对象与数据库中的对象进行比较，找出与输入最相似的对象。索引是有效组织数据的过程，极大地加速了对大型数据集的查询，在相似性搜索的实现中起着重要作用。对一个大规模向量数据集创建索引后，查询可以被路由到最有可能包含与输入查询相似的向量的集群或数据子集。在实践中，这意味着要牺牲一定程度的准确性来加快对真正的大规模向量数据集的查询。


<img align=“center” src="/img/milvus.png"/>

以文搜图的整体流程如上图所示，图片来自[链接](https://github.com/milvus-io/bootcamp/blob/master/solutions/image/text_image_search/workflow.png)：

1. 构建图像特征库

> 图像数据集通过多模态模型提取特征，本文中使用中文CLIP多模态模型，然后将特征存储到Milvus中，用于后续的检索。

2. 文本检索

> 请求文本通过多模态模型提取特征，得到文本特征，然后从Milvus数据库中进行检索，得到topK个相似的内容。

## **Milvus 应用场景**

你可以使用 Milvus 搭建符合自己场景需求的向量相似度检索系统。Milvus 的使用场景如下所示：

- **[图片检索系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/image_similarity_search.md)**：以图搜图，从海量数据库中即时返回与上传图片最相似的图片。
- **[视频检索系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/video_similarity_search.md)**：将视频关键帧转化为向量并插入 Milvus，便可检索相似视频，或进行实时视频推荐。
- **[音频检索系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/audio_similarity_search.md)**：快速检索海量演讲、音乐、音效等音频数据，并返回相似音频。
- **[分子式检索系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/molecular_similarity_search.md)**：超高速检索相似化学分子结构、超结构、子结构。
- **[推荐系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/recommendation_system.md)**：根据用户行为及需求推荐相关信息或商品。
- **[智能问答机器人](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/question_answering_system.md)**：交互式智能问答机器人可自动为用户答疑解惑。
- **[DNA 序列分类系统](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/dna_sequence_classification.md)**：通过对比相似 DNA 序列，仅需几毫秒便可精确对基因进行分类。
- **[文本搜索引擎](https://link.zhihu.com/?target=https%3A//milvus.io/cn/docs/v2.0.x/text_search_engine.md)**：帮助用户从文本数据库中通过关键词搜索所需信息。

---

**本地安装**PyMilvus

> - **Python 版本大于 3.6**
> - `pip3 install protobuf==3.20.0`
> - `pip3 install grpcio-tools`
> - **pip install pymilvus==2.2.3**

下面用脚本hello_milvus.py演示PyMilvus的基本操作。

> - \# 1. 连接向量数据库 **Milvus**
> - \# 2. 创建数据集合 **collection**
> - \# 3. 插入数据实体 **entities**
> - \# 4. 创建索引 **index**
> - \# 5. 搜索、查询
> - \# 6. 删除数据实体
> - \# 7. 删除数据集合

```javascript
import time
import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="******", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `hello_milvus`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
]

insert_result = hello_milvus.insert(entities)

hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
hello_milvus.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
vectors_to_search = entities[-1][-2:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `random > 0.5`"))

start_time = time.time()
result = hello_milvus.query(expr="random > 0.5", output_fields=["random", "embeddings"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# pagination
r1 = hello_milvus.query(expr="random > 0.5", limit=4, output_fields=["random"])
r2 = hello_milvus.query(expr="random > 0.5", offset=1, limit=3, output_fields=["random"])
print(f"query pagination(limit=4):\n\t{r1}")
print(f"query pagination(offset=1, limit=3):\n\t{r2}")


# -----------------------------------------------------------------------------
# hybrid search
print(fmt.format("Start hybrid searching with `random > 0.5`"))

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
ids = insert_result.primary_keys

expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

hello_milvus.delete(expr)

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format("Drop collection `hello_milvus`"))
utility.drop_collection("hello_milvus")
```

---

## 搭建CLIP搜索服务



## 安装基础工具包

我们用到了以下工具：

- **Towhee** 用于构建模型推理流水线的框架，对于新手非常友好。
- **Faiss** 高效的向量近邻搜索库。
- **Gradio** 轻量级的机器学习 Demo 构建工具。

创建一个 conda 环境

```bash
conda create -n img_retrieval python=3.9 
conda activate img_retrieval 
```

安装依赖

```
pip install towhee gradio 
conda install -c pytorch faiss-cpu
```

准备图片库数据

<img align=“center” src="/img/clipimg.png"/>

我们选取 ImageNet 数据集的子集作为本文所使用的 “小型宠物图片库”。首先，下载数据集并解压：

```bash
curl -L -O https://github.com/towhee-io/examples/releases/download/data/pet_small.zip
unzip -q -o pet_small.zip
```
---
以上主要介绍了Milvus的以文搜图。

---
## 相关阅读
- [CLIP模型-中文](https://www.modelscope.cn/models/damo/multi-modal_clip-vit-base-patch16_zh/summary)
- [Milvus 使用手册](https://www.bookstack.cn/read/Milvus/userguide-milvus_operation.md#%E8%BF%90%E8%A1%8C%E6%93%8D%E4%BD%9C%E5%89%8D%E7%9A%84%E5%87%86%E5%A4%87)
- [github--Chinese-CLIP](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2FOFA-Sys%2FChinese-CLIP)

