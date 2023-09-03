# 语音合成学习（十七）学习笔记

---

## SafeTensors简介

"SafeTensors" 是一个文件格式，用于存储和传输机器学习模型中的张量数据。它的设计目的是确保数据的安全性和可靠性，并提供了一些额外的功能特性。

Safetensors结合使用高效的序列化和压缩算法来减少大型张量的大小，使其比pickle等其他序列化格式更快、更高效。这意味着，与传统PyTorch序列化格式pytorch_model.bin和model.safetensors相比，Safetensors在CPU上的速度快76.6倍，在GPU上的速度快2倍。



**以下是 SafeTensors 文件格式的简单介绍：**

1. 安全性：SafeTensors 文件格式使用加密算法来保护数据的安全性。通过加密，可以确保在数据传输和存储过程中，数据不会被未经授权的人或程序访问或篡改。这对于涉及敏感信息或保密模型的应用非常重要。
2. 兼容性：SafeTensors 文件格式与多种计算平台和框架兼容。它提供了可以在不同环境中进行读写操作的库和工具。这意味着您可以在各种机器学习框架和设备上轻松共享和使用 SafeTensors 格式的数据。
3. 元数据：SafeTensors 文件格式支持存储元数据。元数据是关于张量数据的描述信息，如形状、数据类型、名称等。这些元数据可以帮助您更好地理解和处理数据，以及正确地加载和使用模型。
4. 压缩和优化：SafeTensors 文件格式还支持数据的压缩和优化。通过压缩技术，文件的大小可以减小，从而节省存储空间和提高数据传输效率。此外，还可以应用一些优化技术，如量化和稀疏表示，以加速模型加载和推理等过程。
5. 扩展性：SafeTensors 文件格式具有良好的扩展性，可以方便地支持新的功能和特性。这使得它能够适应不断发展的机器学习需求，并且为未来的创新提供了灵活性和可扩展性。

总而言之，SafeTensors 文件格式是一种旨在保护数据安全并提供额外功能的文件格式。它通过加密、兼容性、元数据支持、压缩优化和扩展性等特性，为机器学习数据的存储和传输提供了一个可靠和高效的解决方案。

**使用Safetensors**

介绍safetensors API，以及如何保存和加载张量文件。可以使用pip管理器安装safetensors：

### 一、安装

#### 1.1 pip 安装

```python
pip install safetensors
```

#### 1.2 conda 安装

```python
conda install -c huggingface safetensors
```

### 二、加载张量

本文将使用Torch共享张量中的示例来搭建一个简单的神经网络，并使用PyTorch的safetensors.torch API保存模型。

```python
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = self.a
def forward(self, x):
    return self.b(self.a(x))
   model = Model()
print(model.state_dict())正如所看到的，已经成功创建了模型。
```

```python
OrderedDict([('a.weight', tensor([[-0.0913, 0.0470, -0.0209, ..., -0.0540, -0.0575, -0.0679], [ 0.0268, 0.0765, 0.0952, ..., -0.0616, 0.0146, -0.0343], [ 0.0216, 0.0444, -0.0347, ..., -0.0546, 0.0036, -0.0454], ...,
```

现在我们将通过提供model对象和文件名来保存模型，然后把保存的文件加载到使用nn.Module创建的model对象中。

```python
from safetensors.torch import load_model, save_model

save_model(model, "model.safetensors")

load_model(model, "model.safetensors")
print(model.state_dict())


OrderedDict([('a.weight', tensor([[-0.0913, 0.0470, -0.0209, ..., -0.0540, -0.0575, -0.0679], [ 0.0268, 0.0765, 0.0952, ..., -0.0616, 0.0146, -0.0343], [ 0.0216, 0.0444, -0.0347, ..., -0.0546, 0.0036, -0.0454], ...,
```

在第二个示例中，我们将尝试保存使用torch.zeros创建的张量，为此将使用save_file函数。

```python
import torch
from safetensors.torch import save_file, load_file

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "new_model.safetensors")
```

为了加载张量，我们将使用load_file函数。

```python
load_file("new_model.safetensors")
```

```python
{'weight1': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]),
 'weight2': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]])}
```

Safetensors API适用于Pytorch、Tensorflow、PaddlePaddle、Flax和Numpy，可以通过阅读Safetensors文档来了解它。

---

Checkpoint和safetensors是Stable Diffusion中常见的两种模型存储格式。


Checkpoint

Checkpoint是Tensorflow框架中常用的模型保存方式。它保存了模型的权重和优化器的状态，以便恢复训练。在Stable Diffusion中，Checkpoint格式的模型可以包含更多的训练信息，包括训练过程中的中间状态。这意味着Checkpoint格式的模型文件通常较大。

Checkpoint格式的模型适合需要恢复训练的情况。它们有助于调整和优化模型，因为我们可以看到模型训练过程中的各种信息。

safetensors

safetensors是一种相对较新的模型存储格式，专门为Stable Diffusion模型设计。它的特点是可以存储大型模型，同时保持文件的小巧和快速加载。safetensors只保存模型的权重，而不包含优化器状态或其他信息。这意味着它通常用于模型的最终版本，当我们只关心模型的性能，而不需要了解训练过程中的详细信息时，这种格式是一个很好的选择。由于它的加载速度快，因此更适合实时应用，如在线服务。

在选择模型的存储格式时，需要根据使用场景来决定。例如，如果你需要进行模型微调，或者需要在训练过程中获得详细的信息，Checkpoint可能是更好的选择。而对于那些仅需要快速加载和执行模型的场景，safetensors可能是更好的选择。


---

## 相关资源

> [safetensors的git地址](https://github.com/huggingface/safetensors) 
>
> [safetensors的huggingface地址]()
