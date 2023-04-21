# transformers学习（七）学习笔记

---

Transformer 是一种基于自注意力机制的神经网络模型。它的设计灵感来自于大脑神经元的连接方式，旨在模拟人类大脑的处理过程。Transformer 模型由编码器和解码器组成，编码器将输入序列编码为向量，而解码器将该向量转换为目标序列。

Transformer 的核心思想是自注意力机制。在自注意力机制中，每个位置的注意力权重取决于周围的位置和自身的历史信息。这种机制使得 Transformer 能够捕捉序列中的上下文信息，从而提高了模型的性能。

Transformer 还有许多优化技巧。例如，BERT 模型采用了预训练和后训练技术，即在训练之前先进行预训练，然后再进行微调和训练。此外，Transformer 还采用了多任务学习技术，即在同一模型中同时训练多个任务，从而提高了模型的泛化能力。

[Transformers](https://huggingface.co/docs/transformers/index) 是由 [Hugging Face](https://huggingface.co/) 开发的一个 NLP 包，支持加载目前绝大部分的预训练模型。

------

# 1. 开箱即用的 pipelines

Transformers 库将目前的 NLP 任务归纳为几下几类：

- **文本分类：**例如情感分析、句子对关系判断等；
- **对文本中的词语进行分类：**例如词性标注 (POS)、命名实体识别 (NER) 等；
- **文本生成：**例如填充预设的模板 (prompt)、预测文本中被遮掩掉 (masked) 的词语；
- **从文本中抽取答案：**例如根据给定的问题从一段文本中抽取出对应的答案；
- **根据输入文本生成新的句子：**例如文本翻译、自动摘要等。

Transformers 库最基础的对象就是 `pipeline()` 函数，它封装了预训练模型和对应的前处理和后处理环节。我们只需输入文本，就能得到预期的答案。目前常用的 [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) 有：

- `feature-extraction` （获得文本的向量化表示）
- `fill-mask` （填充被遮盖的词、片段）
- `ner`（命名实体识别）
- `question-answering` （自动问答）/（阅读理解）
- `sentiment-analysis` （情感分析）
- `summarization` （自动摘要）
- `text-generation` （文本生成）
- `translation` （机器翻译）
- `zero-shot-classification` （零训练样本分类）

模型示例：

```python3
from transformers import pipeline
question_answerer = pipeline("question-answering")
```

第二步，将文章和问题输入pipeline，得到预测结果

```python3
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
```

接下来的步骤与前面类型，但是在调用pipeline时，需要将模型和分词器进行指定，代码如下：

```python3
zh_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
QA_input = {'question': "著名诗歌《假如生活欺骗了你》的作者是",'context': "普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。"}
zh_qa(QA_input)

{'score': 0.976427278518677,'start':0,'end':3,'answer':'普希金'}
```

#### pipeline 背后做了什么：

1. 预处理 (preprocessing)，将原始文本转换为模型可以接受的输入格式；
2. 将处理好的输入送入模型；
3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。


<img align=“center” src="/img/pipeline.png"/>


### 使用分词器进行预处理

因为神经网络模型无法直接处理文本，因此首先需要通过**预处理**环节将文本转换为模型可以理解的数字。具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行：

1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 **tokens**；
2. 根据模型的词表将每个 token 映射到对应的 token 编号（就是一个数字）；
3. 根据模型的需要，添加一些额外的输入。

我们对输入文本的预处理需要与模型自身预训练时的操作完全一致，只有这样模型才可以正常地工作。注意，每个模型都有特定的预处理操作，如果对要使用的模型不熟悉，可以通过 [Model Hub](https://huggingface.co/models) 查询。这里我们使用 `AutoTokenizer` 类和它的 `from_pretrained()` 函数，它可以自动根据模型 checkpoint 名称来获取对应的分词器。

情感分析 pipeline 的默认 checkpoint 是 [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)，下面我们手工下载并调用其分词器：

```
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
raw_inputs = [    "I've been waiting for a HuggingFace course my whole life.",    "I hate this so much!",
] 
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") print(inputs)
```

```
{    
'input_ids': tensor([      
		[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],       
    [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,          
    0,     0,     0,     0,     0,     0]  
  ]),   
 'attention_mask': tensor([    
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],     
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]   
  ])
}
```

可以看到，输出中包含两个键 `input_ids` 和 `attention_mask`，其中 `input_ids` 对应分词之后的 tokens 映射到的数字编号列表，而 `attention_mask` 则是用来标记哪些 tokens 是被填充的（这里“1”表示是原文，“0”表示是填充字符）。

### 将预处理好的输入送入模型

预训练模型的下载方式和分词器 (tokenizer) 类似，Transformers 包提供了一个 `AutoModel` 类和对应的 `from_pretrained()` 函数。下面我们手工下载这个 distilbert-base 模型：

```
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

预训练模型的本体只包含基础的 Transformer 模块，对于给定的输入，它会输出一些神经元的值，称为 hidden states 或者特征 (features)。对于 NLP 模型来说，可以理解为是文本的高维语义表示。这些 hidden states 通常会被输入到其他的模型部分（称为 head），以完成特定的任务，例如送入到分类头中完成文本分类任务。

其实前面我们举例的所有 pipelines 都具有类似的模型结构，只是模型的最后一部分会使用不同的 head 以完成对应的任务。

<img align=“center” src="/img/modules.png"/>

Transformer 模块的输出是一个维度为 (Batch size, Sequence length, Hidden size) 的三维张量，其中 Batch size 表示每次输入的样本（文本序列）数量，即每次输入多少个句子，上例中为 2；Sequence length 表示文本序列的长度，即每个句子被分为多少个 token，上例中为 16；Hidden size 表示每一个 token 经过模型编码后的输出向量（语义表示）的维度。

> 预训练模型编码后的输出向量的维度通常都很大，例如 Bert 模型 base 版本的输出为 768 维，一些大模型的输出维度为 3072 甚至更高。
>
> 我们可以打印出这里使用的 distilbert-base 模型的输出维度：
>
> ```
> from transformers import AutoTokenizer, AutoModel
> 
> checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
> model = AutoModel.from_pretrained(checkpoint)
> 
> raw_inputs = [
>     "I've been waiting for a HuggingFace course my whole life.",
>     "I hate this so much!",
> ]
> inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
> outputs = model(**inputs)
> print(outputs.last_hidden_state.shape)
> torch.Size([2, 16, 768])
> ```

# 2. 模型 Models 

常用的模型一般分为三种：自回归模型、自编码模型和序列到序列模型。

- **自回归模型**采用经典的语言模型任务进行预训练，即给出上文，预测下文，对应原始Transformer模型的解码器部分，其中最经典的模型是GPT。由于自编码器只能看到上文而无法看到下文的特点，模型一般会用于文本生成的任务。
- **自编码模型**则采用句子重建的任务进行预训练，即预先通过某种方式破坏句子，可能是掩码，可能是打乱顺序，希望模型将被破坏的部分还原，对应原始Transformer模型的编码器部分，其中最经典的模型是BERT。与自回归模型不同，模型既可以看到上文信息，也可以看到下文信息，由于这样的特点，自编码模型往往用于自然语言理解的任务，如文本分类、阅读理解等。（此外，这里需要注意，自编码模型和自回归模型的唯一区分其实是在于预训练时的任务，而不是模型结构。）
- **序列到序列模型**则是同时使用了原始的编码器与解码器，最经典的模型便是T5。与经典的序列到序列模型类似，这种模型最自然的应用便是文本摘要、机器翻译等任务，事实上基本所有的NLP任务都可以通过序列到序列解决。

下表中总结了以上三种类型模型的常用预训练模型，以及适合处理的解决的任务。

| 模型类型       | 常用预训练模型                    | 适用任务                         |
| -------------- | --------------------------------- | -------------------------------- |
| 自回归模型     | CTRL, GPT, GPT-2, Transformer XL  | 文本生成                         |
| 自编码模型     | ALBERT, BERT, DistilBERT, RoBERTa | 文本分类、命名实体识别、阅读理解 |
| 序列到序列模型 | BART, T5, Marian, mBART           | 文本摘要、机器翻译               |

以加载bert-base-chinese模型为例，代码如下。

```python3
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-chinese")
```

> `Model.from_pretrained()` 会自动缓存下载的模型权重，默认保存到 *~/.cache/huggingface/transformers*，我们也可以通过 HF_HOME 环境变量自定义缓存目录。

所有存储在 [Model Hub](https://huggingface.co/models) 上的模型都能够通过 `Model.from_pretrained()` 加载，只需要传递对应 checkpoint 的名称。当然了，我们也可以先将模型下载下来，然后将本地路径传给 `Model.from_pretrained()`，比如加载下载好的 [Bert-base 模型](https://huggingface.co/bert-base-cased)：

```python
from transformers import BertModel

model = BertModel.from_pretrained("./models/bert/")
```

部分模型的 Hub 页面中会包含很多文件，我们通常只需要下载模型对应的 *config.json* 和 *pytorch_model.bin*，以及分词器对应的 *tokenizer.json*、*tokenizer_config.json* 和 *vocab.txt*。

### 保存模型

保存模型与加载模型类似，只需要调用 `Model.save_pretrained()` 函数。例如保存加载的 BERT 模型：

```
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/")
```

这会在保存路径下创建两个文件：

- config.json：模型配置文件，里面包含构建模型结构的必要参数；
- pytorch_model.bin：又称为 state dictionary，包含模型的所有权重。

这两个文件缺一不可，配置文件负责记录模型的**结构**，模型权重记录模型的**参数**。我们自己保存的模型同样可以通过 `Model.from_pretrained()` 加载，只需要传递保存目录的路径。

# 3.分词器  Tokenizer

因为神经网络模型不能直接处理文本，我们需要先将文本转换为模型能够处理的数字，这个过程被称为**编码 (Encoding)**：先使用分词器 (Tokenizers) 将文本按词、子词、符号切分为 tokens；然后将 tokens 映射到对应的 token 编号（token IDs）。

（1）分词：使用分词器对文本数据进行分词（字、字词）；

（2）构建词典：根据数据集分词的结果，构建词典映射（这一步并不绝对，如果采用预训练词向量，词典映射要根据词向量文件进行处理）；

（3）数据转换：根据构建好的词典，将分词处理后的数据做映射，将文本序列转换为数字序列；

（4）数据填充与截断：在以batch输入到模型的方式中，需要对过短的数据进行填充，过长的数据进行截断，保证数据长度符合模型能接受的范围，同时batch内的数据维度大小一致。

在以往的工作中，我们可能会使用不同的分词器，并自行实现构建词典与转换的工作。但是在transformers工具包中，无需再这般复杂，只需要借助Tokenizer模块便可以快速的实现上述全部工作，它的功能就是将文本转换为神经网络可以处理的数据。Tokenizer工具包无需额外安装，会随着transformers一起安装。下面，演示一下具体该如何使用Tokenizer。

#### 加载分词器

```python3
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
```

句子分词

使用tokenize方法进行分词，代码如下。可以看到，bert-base-chinese分词是按照字来分的。

```python3
sen = "弱小的我也有大梦想"
tokens = tokenizer.tokenize(sen)
tokens

结果：['弱','小','的','我','也','有','大','梦','想']
```

### 查看词典

分词之后，应该构建词典，但是正如前面所言，Tokenzier是随着预训练模型一起产生的，因此词典已经预先构建好了，无需再次构建。关于词典的具体内容，可以通过vocab进行查看。

```
Tokenizer.vocab

结果：
'行'： 6121，
‘貂’：6503
...
```

### **词序列转数字序列**

Tokenizer提供了更加便捷的encode方法：

```text
ids = tokenizer.encode(sen)
ids

结果：【233，101，102，2207，738，1902】
```

### 填充与截断

借助encode方法，还可以很方便的做到对数据的填充与截断，只需要我们指定对应的参数即可，代码如下：

```python3
# 填充
ids = tokenizer.encode(sen, padding="max_length", max_length=10)
ids

结果：【233，101，102，2207，738，1902，0，0，0，0】
```

```python3
# 裁剪
ids = tokenizer.encode(sen, max_length=5, truncation=True)
ids

结果：【233，101，102，2207，738】
```

### **attention_mask 与 token_type_id**

还没结束，数据要能够输入transformers提供的预训练模型，还需要构建attention_mask和token_type_id这两个额外的输入，分别用于标记真实的输入与片段类型，我们可以通过下面这段代码实现

```python3
ids = tokenizer.encode(sen, padding="max_length", max_length=15)
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
attention_mask, token_type_ids

结果：
(【1，1，1，1，1，1，0，0，0，0】,
【0，0，0，0，0，0，0，0，0，0】)

'token_type_ids'， 区分两个句子的编码
'attention_mask', 指定对哪些词进行self-Attention操作
```

---

>huggingface的官方网站：[http://www.huggingface.co.](https://link.zhihu.com/?target=http%3A//www.huggingface.co./) 在这里主要有以下大家需要的资源。
>
>1. Datasets：数据集，以及数据集的下载地址
>2. Models：各个预训练模型
>3. course：免费的nlp课程，可惜都是英文的
>4. docs：文档
>
>[huggingface官方教程](https://huggingface.co/docs/transformers/model_doc/bert)
>
>[安装Huggface库](https://huggingface.co/transformers/installation.html#)(需要预先安装pytorch)

------

# Task

transformers 预训练的任务包括语言建模、掩码语言建模、下一句预测、连续文本预测、文本分类、问答等任务。

1. 语言模型预训练：例如BERT、GPT、ELMo等，该任务的目标是让模型学习到一个通用的语言表示形式，通常采用掩码语言模型（Masked Language Modeling）和下一句预测（Next Sentence Prediction）等技术。
2. 问答系统预训练：例如T5、UniLM等，该任务的目标是让模型能够根据输入的问题和上下文生成对应的答案，通常使用自然语言生成和问答匹配等技术。
3. 序列标注预训练：例如BERT-CRF、Transformer-CRF等，该任务的目标是让模型学习到序列的内部结构和标签，通常采用条件随机场（CRF）等技术。
4. 机器翻译预训练：例如Marian、Transformer-XL等，该任务的目标是让模型能够学习到翻译的规律和模式，通常采用注意力机制和自回归模型等技术。
5. 图像生成预训练：例如DALL-E、CLIP等，该任务的目标是让模型能够将自然语言描述转化为相应的图像或视频，通常使用生成对抗网络（GAN）等技术。

------

### AutoModel

>是一种自动化预训练模型选择工具，其能够根据输入的任务和语言数据类型自动选择适合的预训练模型，包括BERT、GPT、RoBERTa、XLM等等，并进行相应的配置，使用户可以更加方便地使用Transformers预训练模型。
>
>使用AutoModel，用户只需要指定任务名称（如分类、序列标注等）和输入数据类型（如文本、图像等），AutoModel就会自动选择对应的预训练模型，并提供了一些自定义参数选项，以满足用户特定的需求。另外，AutoModel也支持从预训练模型中选择指定的层或者几何自动编码器中的几何表示层。它的设计使得使用Transformers平台更加简单，减少了手动选择预训练模型和调参的复杂性和工作量。



### AutoModelForCausalLM

> Transformers预训练的AutoModelForCausalLM是一种用于生成式语言模型（Generative Language Model）的自动化预训练模型选择工具。预测一个sequence之后的token的任务，在这种情境下，模型只会attend left context（mask左边的token）。这样的训练设置特别关注于生成任务。一般来说，预测下一个token是通过抽样输入sequence得到的最后一层hidden state的logits得到的。
>
> 该模型根据输入的任务和语言数据类型自动选择适合的预训练模型，例如GPT和GPT-2等模型，并进行相应的配置。AutoModelForCausalLM的目标是为用户提供一个简单易用的API，以生成具有连续性的文本。
>
> 与其他生成模型相比，它的优势在于它是一种基于自回归（Autoregressive）的方法，可以使用前一时刻预测后一时刻，由此生成连续性文本。 所以，AutoModelForCausalLM主要用于基于自回归的生成式任务，例如文本生成及其它生成式任务。此模型对于诸如电影脚本生成，机器翻译，画像生成，音乐生成等用例非常有用。

### AutoModelForImageClassification

> AutoModelForImageClassification是使用Hugging Face Transformers库中的自动模型选择功能，根据输入的数据集和任务类型自动选择预训练图像分类模型。它可以处理不同的输入数据类型和任务类型，包括图像分类、目标检测和图像分割等。具体而言，AutoModelForImageClassification会自动选择并加载最适合输入数据和任务类型的预训练模型，并通过对齐预训练模型和目标任务，进行微调以完成图像分类任务。它可以提高模型的性能，同时减少了手动选择和配置模型的工作量。

### AutoModelForImageSegmentation

>  AutoModelForImageSegmentation是使用Hugging Face Transformers库中的自动模型选择功能，根据输入的数据集和任务类型自动选择预训练图像分割模型。它可以处理不同的输入数据类型和任务类型，包括语义分割、实例分割和全景分割等。具体而言，AutoModelForImageSegmentation会自动选择并加载最适合输入数据和任务类型的预训练模型，然后使用卷积神经网络对输入的图像进行特征提取，并生成与输入图像相同大小的分割图像，其中每个像素都被分配到预定义的标签。通过对齐预训练模型和目标任务，进行微调以完成图像分割任务。它可以提高模型的性能，同时减少了手动选择和配置模型的工作量。



### AutoModelForMaskedImageModeling

> AutoModelForMaskedImageModeling是一个预训练模型，用于执行掩膜图像建模任务。掩膜图像建模是指将输入的图像中的某些部分用掩膜进行覆盖，然后尝试预测被覆盖的部分。这个任务可以用于图像修复、电影特效、文本识别等。
> AutoModelForMaskedImageModeling基于最先进的掩膜图像建模模型，如Mask R-CNN，DETR-RCNN等。它还支持在多种预训练的transformers模型上进行训练，例如Bert，Roberta，Electra，等等。



### AutoModelForMaskedLM

> AutoModelForMaskedLM是一个预训练模型，用于执行掩膜语言建模任务。掩膜语言建模是指将输入的句子中的一些单词使用掩膜进行覆盖，然后尝试预测被覆盖的单词。这个任务可以用于许多自然语言处理任务，例如文本生成、机器翻译、自动摘要等。
>
> 它可以根据输入的数据和任务要求自动选择合适的Masked Language Modeling (MLM) 模型进行预训练，并在模型的基础上进行微调或者训练。
>
> Masked Language Modeling 用masking token来mask sequence中的一些tokens，然后调整模型使之用合适的token来填充这些mask。这让模型能够attend right context（mask右边的token）和left context（mask左边的token）。这样的训练设置为需要bi-directional context的下游任务（如SQuAD1）提供了强基础。
>
> AutoModelForMaskedLM使用了Hugging Face Transformers库中的transformers.AutoModelForPreTraining和transformers.AutoConfig类。这些类可以自动选择在预训练时使用的Transformer模型，并生成与数据和任务匹配的配置文件。
>
> 具体来说，AutoModelForMaskedLM可以选择以下常见的MLM模型：
>
> BERT	RoBERTa	ALBERT	DistilBERT	ELECTRA ...
>
> 这些模型都是使用基于Transformers的encoder-decoder架构。其中BERT是最早提出的MLM模型，后来出现的RoBERTa和ALBERT在BERT的基础上进行改进，基本上都具有比BERT更好的性能。DistilBERT是一个轻量级的BERT模型，可以在保持性能的同时大大减少模型大小和计算量。ELECTRA是一种基于对抗训练的MLM模型，相对于传统的MLM模型，它可以更快地进行训练，同时在一定程度上提高训练效果。



### AutoModelForMultipleChoice

>该类可根据提供的预训练模型自动构建多选题任务模型，支持多种预训练模型，包括BERT、RoBERTa、DistilBERT等。这些模型的灵感来自于语言模型，通过将问题和每个候选答案结合起来，将其编码为向量。然后，在这些向量的基础上执行二元分类的任务，以确定最佳答案。
>
>使用AutoModelForMultipleChoice，用户可以更方便地训练自己的多选题模型，过程中也省去了从预训练模型手动构建模型的步骤。



### AutoModelForObjectDetection

> AutoModelForObjectDetection是一个自动化的模型选择器，它可以根据输入的数据和任务要求自动选择合适的目标检测模型进行预训练，并在模型的基础上进行微调或者训练。
>
>AutoModelForObjectDetection使用了Hugging Face Transformers库中的transformers.AutoModelForPreTraining和transformers.AutoConfig类。这些类可以自动选择在预训练时使用的Transformer模型，并生成与数据和任务匹配的配置文件。
>
>具体来说，AutoModelForObjectDetection可以选择以下常见的目标检测框架：
>
>Faster R-CNN	Mask R-CNN	RetinaNet	YOLOv3	EfficientDet ...
>同时，还可以选择各种基于Transformers的backbone，如BERT、RoBERTa、DistilBERT等。



### AutoModelForQuestionAnswering

>AutoModelForQuestionAnswering是一个基于transformers预训练模型的工具，它的作用是为给定的问答任务选择最合适的预训练模型，并对其进行微调以适应特定任务。
>
>具体来说，AutoModelForQuestionAnswering将输入的问题（question）和一个可能包含答案的文本段落（context）进行处理，然后使用如RoBERTa、Bert等transformers预训练模型的结构对问题和文本段落进行编码，最终输出一个包含答案的文本片段。在训练过程中，AutoModelForQuestionAnswering使用了如softmax等损失函数来对模型参数进行更新。
>
>AutoModelForQuestionAnswering涉及的预训练模型包括BertForQuestionAnswering、RobertaForQuestionAnswering和DistilBertForQuestionAnswering等。这些模型由于在自然语言处理领域取得的显著成果，被广泛应用于问答系统、机器翻译、摘要生成、语言理解等多个领域。
>
>AutoModelForQuestionAnswering可以帮助开发者更加高效地构建高性能的问答系统，降低模型开发和优化的成本。



### AutoModelForSemanticSegmentation

> AutoModelForSemanticSegmentation是一个预训练模型，用于执行语义分割任务。它使用了自动化模型选择的技术，基于给定的任务、语言和其他超参数自动选择最佳的模型。该模型的输入是图像，输出是每个像素的类别标签，它可以将像素分为不同的区域，例如“人”，“车”等。旨在将图像中的每个像素与其对应的物体或场景类别关联起来
>      
>AutoModelForSemanticSegmentation基于最先进的语义分割模型，如U-Net，SegNet，DeepLab等。它还支持在多种预训练的transformers模型上进行训练，例如Bert，Roberta，Electra，等等。
>
>
>该类可根据提供的预训练模型自动构建语义分割模型，支持多种预训练模型，包括UNet、DeepLabV3、PSPNet等。这些模型旨在通过编码图像特征来推断每个像素包含的类别标签，并且在许多视觉任务和数据集上都取得了出色的效果。
>
>使用AutoModelForSemanticSegmentation，用户可以更方便地训练自己的语义分割模型，过程中也省去了从预训练模型手动构建模型的步骤。



### AutoModelForSeq2SeqLM

> AutoModelForSeq2SeqLM是一个在给定的文本数据集上进行序列到序列生成任务的预训练模型选择工具。它可以自动选择适合给定数据集的最佳预训练seq2seq模型，并对其进行微调以适应特定的序列到序列生成任务。
>
>具体来说，它将输入的文本数据进行tokenization（标记化），并使用encoder-decoder的序列到序列模型（如T5、Bart、Pegasus等）对输入序列进行编码，然后根据生成任务的类型（如翻译、摘要、问答等）对输出序列进行解码生成。在训练过程中，它使用了如beam search等技术对模型进行优化，并使用cross entropy等损失函数对模型参数进行更新。AutoModelForSeq2SeqLM可以帮助开发者更快地找到适合自己应用场景的seq2seq模型，从而加速模型开发和部署。

### AutoModelForSequenceClassification

>    AutoModelForSequenceClassification是一个在给定的文本数据集上进行分类任务的预训练模型选择的工具。它可以自动根据给定的数据集选择最适合的预训练模型，自动构建一个适用于序列分类任务的模型,并对该模型进行微调，以适应特定的分类任务。
>    
>    具体来说，它将输入的文本数据处理为token embeddings，然后将这些embeddings输入到BERT、RoBERTa、XLNet、DistilBERT、ALBERT等常见的预训练模型中，然后将最后一个隐藏层的表示用于分类任务。在训练过程中，它使用了交叉熵作为损失函数，使用优化器对模型参数进行更新。该模型可以接受输入序列并输出该序列对应的标签。
>    
>    使用AutoModelForSequenceClassification，用户可以更容易地训练自己的序列分类模型，省去了手动从预训练模型构建模型的麻烦。
>    
>    使用AutoModelForSequenceClassification的一些常见的预训练模型包括BERT、RoBERTa、XLNet和DistilBERT等。

### AutoModelForSpeechSeq2Seq

>AutoModelForSpeechSeq2Seq是一个基于预训练的自动模型，用于将输入的语音序列转换为文本序列。具体来说，它使用序列到序列（Seq2Seq）神经网络来学习将输入语音序列映射到相应的文本序列。用于进行语音识别和语音翻译等语音序列转换任务。
>
>需要注意的是，AutoModelForSpeechSeq2Seq需要使用特定的输入数据格式，即音频文件和相应的标签文件。音频文件应该是wav格式，标签文件应该是与音频文件相同长度的文本序列，可以使用拼音或字符表示。同时，还需要进行数据预处理和数据增强操作，以提高模型的性能和泛化能力。
>
>使用AutoModelForSpeechSeq2Seq可以极大地简化语音识别任务的实现过程，使得开发者无需手动构建复杂的模型和特征工程流程，只需将原始语音数据输入模型即可获得对应的文本输出。
>
>AutoModelForSpeechSeq2Seq使用的具体模型包括：Wav2Vec2、S2T、M2M等。其中，Wav2Vec2是基于自监督的语音特征提取模型，旨在提高语音识别任务的性能。S2T是一种将语音序列转换为文本序列的模型，可以用于语音转写的任务。M2M则是一种将语音序列转换为语音序列（翻译）的模型，可以用于跨语言的语音翻译任务。

### AutoModelForTokenClassification

>AutoModelForTokenClassification是一个神经网络模型，基于预训练的自动模型，用于将输入的文本序列中的每个标记分类为预定义的标记类型。具体来说，它将输入序列中的每个标记丢给模型进行预测，以确定其属于哪个标记类型。
>
> 可以进行命名实体识别（Named Entity Recognition）任务。用于进行命名实体识别和序列标注任务。它能够自动识别输入句子中的词汇，并对其进行分类，判断其属于哪种类型的实体（如人名、组织机构、时间等）。它是基于transformers预训练的模型和PyTorch深度学习框架构建的，根据任务需要自动选择最佳的预训练模型，对预训练模型进行微调，进一步提高在具体任务中的性能。
>
>  AutoModelForTokenClassification使用了Hugging Face Transformers库中的transformers.AutoModelForTokenClassification和transformers.AutoConfig类。这些类可以自动选择在预训练时使用的Transformer模型，并生成与数据和任务匹配的配置文件。
>
>  具体来说，AutoModelForTokenClassification可以选择以下常见的Token Classification模型：
>
>​        BERT ：是谷歌开发的一种双向转换编码器，在各种NLP任务上表现出色。
>  RoBERTa ：是由Facebook团队优化的BERT模型，提高了在一些NLP任务上的表现。
>​        XLNet ：是由CMU和Google开发的一种自许多NLP任务上表现出色。
> DistilBERT ：是由Hugging Face开发的一种轻量级的BERT变体，具有较短的训练时间和较小的模型大小，但					   在多NLP任务上表现良好。
>​     ALBERT ：由Google开发的一种轻量级BERT变体，采用了一种参数共享策略，可以在一些NLP好的性能。
>   ELECTRA ：是一种新型的预训练思路，用于文本分类、生成、QA等任务。
>
>这些模型都是使用基于Transformers的encoder-decoder架构，其中BERT是最早提出的模型，后来出现的RoBERTa和ALBERT在BERT的基础上进行改进，基本上都具有比BERT更好的性能。
>XLNet是一种新型的自回归模型，可以在序列标注任务中提供更好的性能。XLM旨在解决跨语种自然语言理解问题，可以在多语种序列标注任务中提供更好的性能。
>DistilBERT是一个轻量级版本的BERT模型，可以在保持性能的同时大大减少模型大小和计算量，通过减少模型大小和计算负载来提高效率和速度。
>ELECTRA也是一种基于对抗训练的模型，相对于传统的Token Classification模型，它可以更快地进行训练，同时在一定程度上提高训练效果。这些模型都可以用于命名实体识别（Named Entity Recognition）、词性标注（Part-of-Speech Tagging）、语义角色标注（Semantic Role Labeling）等任务。

### AutoModelForVision2Seq

> AutoModelForVision2Seq是一个神经网络模型，主要用于图像识别和生成自然语言描述任务。它能够将输入的图像转化为对应的文字描述输出。
>
>该模型通常基于transformers预训练模型和PyTorch深度学习框架构建，可以通过对预训练模型进行微调来适应特定领域或数据集。它能够学习到图像与对应自然语言描述之间的映射关系，并对未知的图像进行自动描述，从而实现图像自动生成文字描述的功能。
>
>在一些应用场景中，例如图片搜索、自动图像描述生成、图像片段组合等，在图像理解与自然语言处理两个领域的交叉点上，AutoModelForVision2Seq有着广泛的应用价值。 

------

## 举一些例子

**AutoModelForSequenceClassification**

 Sequence Classification任务是将sequence在给定的类数中进行分类。如GLUE数据集。

用AutoClass判断两句话是否同义（互为改写）的示例：

- 根据checkpoint名初始化tokenizer和模型，模型架构是BERT，并加载checkpoint中的权重
- 构建一个由两句话组成的sequence，含有正确的model-specific separators, token type ids and attention masks（由tokenizer自动生成）
- 将这个sequence传入模型，对它进行分类：是否同义
- 计算输出的softmax结果，获得在各类上的概率值
- 打印结果

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
```

输出：

```
`not paraphrase: 10%`
`is paraphrase: 90%`
not paraphrase: 94%`
`is paraphrase: 6%
```

------

### **AutoModelForQuestionAnswering**

从context（一段文本）中抽取句子，作为特定问题答句。如SQuAD[1](https://blog.csdn.net/PolarisRisingWar/article/details/123575883#fn1)数据集。

用AutoClass的示例：

- 根据checkpoint名初始化tokenizer和模型，模型架构是BERT，并加载checkpoint中的权重
- 定义context和一些问题
- 迭代所有问题，构建context和当前问题的sequence（用正确的model-specific separators, token type ids and attention masks）
- 将这个sequence传入模型，输出整个sequence上每个token的得分（该token是start index或end index的可能性得分）。
- 计算输出的softmax结果，获得在各token上的概率值
- 获取被识别为start和end之间的值的token，将其转化为字符串
- 打印结果

```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""
questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}")
```

输出：

```
Question: How many pretrained models are available in 🤗 Transformers?
Answer: over 32 +
Question: What does 🤗 Transformers provide?
Answer: general - purpose architectures
Question: 🤗 Transformers provides interoperability between which frameworks?
Answer: tensorflow 2. 0 and pytorch
```

### Language Modeling

Language modeling是使模型适应某一语料（一般是特定领域的）的任务

###  AutoModelForMaskedLM

Masked Language Modeling 用masking token来mask sequence中的一些tokens，然后调整模型使之用合适的token来填充这些mask。这让模型能够attend right context（mask右边的token）和left context（mask左边的token）。这样的训练设置为需要bi-directional context的下游任务（如SQuAD1）提供了强基础。

用AutoClass的示例：

- 根据checkpoint名初始化tokenizer和模型，模型架构是DistilBERT，并加载checkpoint中的权重
- 定义含有一个masked token的sequence：用tokenizer.mask_token（这是个字符串格式的变量，在字符串中用花括号括起来以实现替换2）替换一个单词（我感觉这里的单词应该指的是一个token）
- 将sequence编码为token IDs的列表，找到masked token在列表中的位置。
- 提取在mask token索引值处的预测值：这个张量和vocabulary有同样的尺寸，其元素值就是分配给每个token的得分。模型认为在给定context下，更有可能是这个masked token的token，会得到更高的分数。
- 用PyTorch的topk方法提取得分最高的5个token。
- 用上述的tokens来替代mask token，打印结果。

```
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

sequence = (
    "Distilled models are smaller than the models they mimic. Using them instead of the large "
    f"versions would help {tokenizer.mask_token} our carbon footprint."
)

inputs = tokenizer(sequence, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

token_logits = model(**inputs).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#得到分数最高的5个token的索引
#值得注意的是，topk函数默认是根据value经过sort的。参考其函数文档：https://pytorch.org/docs/stable/generated/torch.topk.html

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    #将该token解码为文本形式，替代原文中的tokenizer.mask_token
```

输出：

```
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
```

###  **AutoModelForCausalLM**

 CLM是预测一个sequence之后的token的任务。在这种情境下，模型只会attend left context（mask左边的token）。这样的训练设置特别关注于生成任务。一般来说，预测下一个token是通过抽样输入sequence得到的最后一层hidden state的logits得到的。

一般来说，预测下一个token是通过抽样输入sequence得到的最后一层hidden state的logits得到的。

用AutoClass的示例：用AutoModelForCausalLM、AutoTokenizer和top_k_top_p_filtering()方法，在输入sequence后抽样得到下一个token：

```
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and"

inputs = tokenizer(sequence, return_tensors="pt")
input_ids = inputs["input_ids"]

# get logits of last hidden state
next_token_logits = model(**inputs).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
```

输出：

```
`Hugging Face is based in DUMBO, New York City, and is`
```

------

以上主要是对Transformer 学习的相关的基本知识做个整理总结，希望能帮助到大家。

---

## 在线demo

> -  [Masked word completion with BERT](https://huggingface.co/bert-base-uncased?text=Paris+is+the+[MASK]+of+France)
> - [Text generation with GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+)

## 相关阅读

> - [Transformers中文使用说明](https://github.com/huggingface/transformers/blob/main/README_zh-hans.md)
>- [数据集查找](https://huggingface.co/datasets)
> - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
