# 语音合成学习（十二）学习笔记

---
# transformers文本分类

## 简介

文本分类任务的输入为一段文本，输出为该段文本的标签。

Transformer是一种基于自注意力机制的神经网络模型，通常用于自然语言处理任务。在文本分类任务中，Transformer模型可以将输入文本作为序列输入，其中每个单词或字符都被嵌入到一个矢量空间中，然后将这些词向量送入Transformer网络中进行处理。
在处理过程中，Transformer模型利用自注意力机制来关注输入序列中的不同部分，以便为每个序列位置生成上下文表示。这种自注意力机制使得模型能够在不需要训练词向量或手动设计特征的情况下，自动捕捉上下文中的语义信息。
在文本分类任务中，Transformer模型可以看作是一个编码器（encoder），它将输入序列映射为一个固定长度的向量表示。这个向量代表了输入文本的语义信息，在分类任务中可以用来预测文本所属的类别。

## 环境配置

我们需要先安装对应的环境。

```console
!pip install transformers==4.21.0 datasets evaluate  //
cd ..
pip install  -r requirements.tx
```

## 导入相关包

我们导入全部需要用到的内容，主要包括数据集、评估、模型、分词器、训练五部分。

关于模型，根据之前1.4章节中的任务对应模型，文本分类任务用到的模型是AutoModelForSequenceClassification，对应导入即可，

```python3
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
```

## 加载数据集

首先我们需要准备数据集文件，处理文件：我是用js处理了excel文件并存储为.json文件。

然后可以使用datasets进行加载，data_files文件指定路径 ，cache_dir指定cache路径（这里报错的情况下可以删除cache里文件）：

```python3
dataset = load_dataset("json", data_files="./data.json",cache_dir='cache)
```

```python3
dataset["train"][0]
```

<img align="center" src="/img/labeltitle.png"/>

在这个数据集中，存在着一条空的数据，我们需要过滤一下，可以使用filter方法。

```python3
dataset = dataset.filter(lambda x: x["review"] is not None)
```

## 数据集处理

数据处理的第一步，需要先加载分词器。

分词器使用的模型是哈工大开源的bert模型。

该模型在Models官网的名称为hfl/chinese-bert-wwm-ext，我们可以快速加载。

```python3
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
```

接下来，定义数据处理函数，文本分类任务无需过多的处理，简单做下文本截断即可。

```python3
def process_function(examples):
  tokenized_examples = tokenizer(examples["review"], max_length=64, truncation=True)
  tokenized_examples["labels"] = examples["label"]
  return tokenized_examples
```

需要注意，这里**要将标签映射到labels字段，这样做是因为使用Trainer进行模型训练时，会自动找到labels字段作为标签。**

定义完数据处理函数，便可以使用map方法，对数据集进行处理，不要忘了指定batched参数值为True，这样会加速数据处理。

```python3
tokenized_datasets = datasets.map(process_function, batched=True)
```

这里可能会有读者好奇，数据并没有做填充处理，后面如何以batch输入模型进行训练。事实上，这里做填充处理是没有问题的。但是如果在这个阶段中做填充，会将所有数据的长度都填充到64，如果训练时一个batch中的数据中都是短文本，那数据中将有大量的填充值，影响计算效率。该如何解决这一问题呢？

我们可以让其在取出一个batch的数据之后再根据batch内数据的最大长度进行填充，如果用Dataloader加载数据，我们可以指定collate_fn，但是这个函数需要我们自行实现。在transformers中，我们则可以使用DataCollatorWithPadding，实例化该类并在Trainer中指定即可。

## 构建评估函数

文本分类任务的评估指标有很多，这里以损失函数作为评价指标，直接通过loss进行加载。

该函数接收一个元组`eval_pred`作为输入参数，该元组包含了模型的预测结果和对应的标签。

首先从`eval_pred`元组中获取模型的预测结果和对应的标签，然后使用PyTorch中的`MultiLabelSoftMarginLoss`损失函数计算预测结果和标签之间的损失（loss）。在这里，`reduction='mean'`表示计算平均损失。最后，该函数返回一个字典，其中包含一个键值对`"loss": loss`，其中`loss`表示计算出的平均损失。

总之，该函数的作用是计算模型的损失，用于评估模型的性能。

```python3
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  fn_loss = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
  loss = fn_loss(torch.from_numpy(predictions), torch.from_numpy(labels))
  return {"loss": loss}
```

## 配置训练器

配置训练器之前，需要先加载模型，这里的模型需要和前面分词器的模型一致，这里要指定支持多标签模型名称外，还要指定num_labels参数，值为label值的个数。

```python3
model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=85)
```

下面，设置训练参数，配置学习率为2e-5 ----控制控制模型参数更新的步长；训练时batch大小为32，评估时为batch大小128，训练轮数为20，权重衰减大小为0.01------用于控制模型参数的大小。输出文件夹为model_for_seqclassification-------模型输出目录，用于保存训练好的模型。日志记录的步长为200-------训练过程中每隔多少步打印一次日志，即10个batch记录一次；评估策略为训练完一个step之后进行评估，模型保存策略同上，设置训练完成后加载最优模型，并指定最优模型的评估指标为loss，这个值要和compute_metrics函数中返回值的键匹配，因为我的机器没有最后就没有指定了半精度训练。

```python3
args = TrainingArguments(
  learning_rate=2e-5,
  per_device_train_batch_size=32,
  per_device_eval_batch_size=128,
  num_train_epochs=20,
  weight_decay=0.01,
  output_dir="model_for_seqclassification",
  logging_steps=200,
  evaluation_strategy="steps",
  save_strategy="steps",
  save_steps=200,
  load_best_model_at_end=True,
  metric_for_best_model="loss",
  fp16=False
)
```

设置完训练参数，就可以构建训练器了。第一个参数指定模型，第二个参数指定训练参数，接下来依次指定训练数据集与验证数据集，这里验证数据集使用了测试集，而后指定分词器以及评估函数，最后指定data_collator的值为DataCollatorWithPadding的实例对象。

```python3
trainer = Trainer(
  model,
  args,
  train_dataset=dataset,
  eval_dataset=dataset,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics,
  data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)
```

## 训练与评估

做完上述全部工作后，就可以开始训练啦!

```python3
trainer.train()
```

<img align="center" src="/img/trainer.png"/>

训练完成后，可以看到在model_for_seqclassification文件夹中存在着5个checkpoint文件夹，保存着不同轮次的模型，runs文件夹中则记录着运行日志，可以使用tensorboard进行可视化。

<img align="center" src="/img/checkpoint.png"/>

---
## 相关阅读
- [Transformers分类任务](https://zhuanlan.zhihu.com/p/548336726)

