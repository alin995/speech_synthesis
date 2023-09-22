# 语音合成学习（十八）学习笔记

---

## LLM InternLM-Chat-7B模型、Baichuan-7B-Chat模型简单使用



**以下是 InternLM-Chat-7B模型、Baichuan-7B-Chat模型的简单介绍：**

> InternLM-Chat-7B模型是OpenAI团队通过大规模预训练得到的一个聊天模型。包含面向实用场景的70亿参数基础模型 （InternLM-7B）。模型具有以下特点：
>
> - 使用上万亿高质量预料，建立模型超强知识体系；
> - 通用工具调用能力，支持用户灵活自助搭建流程；
>
> 它可以应用于各种类型的对话任务，包括问答、闲聊、指导等。你可以向这个模型提出问题或者进行对话，并期望得到符合上下文的回答。
>


Baichuan-7B-Chat模型是百度开发的一个聊天模型，同样是基于GPT-3.5模型进行预训练得到的。它也具备类似的功能，可以进行自然语言对话，并根据上下文生成回答。Baichuan-7B-Chat模型的设计目标是提供高质量的对话体验，满足用户在不同领域的需求。

指定GPU运行：

```python
#方式一 （两行必须放在import  torch前面）
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#方式二
#import torch
#torch.cuda.set_device(0)
```

### 1、InternLM-Chat-7B模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)
```

![image-20230922142929235](/Users/wangwenlin/Desktop/img/internlm.png)
<img aligin="center" src="/img/internlm.png" />

1、使用 LMDeploy 完成 InternLM 的一键部署。
首先安装 LMDeploy:

```
python3 -m pip install lmdeploy
```

快速的部署命令如下：

```
python3 -m lmdeploy.serve.turbomind.deploy InternLM-7B /path/to/internlm-7b/model hf
```

在导出模型后，你可以直接通过如下命令启动服务一个服务并和部署后的模型对话

```
python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
```



### 2、Baichuan-7B-Chat模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

```


<img aligin="center" src="/img/baichuan.png" />



创建虚拟环境

```python
conda create -n baichuan-7b python==3.9 -y 

conda activate baichuan-7b
```

安装Baichuan-7B

```py
git clone --recursive https://github.com/baichuan-inc/Baichuan-13B.git; cd Baichuan-13B 

pip install -r requirements.txt
```

启动Baichuan-7B

```python
cp web_demo.py web_ui.py
```

---

## 相关资源

> [internlm-chat-7b git地址](https://github.com/InternLM/lmdeploy.git) 
>
> [internlm-chat-7b模型下载](https://huggingface.co/internlm/internlm-7b/tree/main)
>
> [Baichuan-7B-Chat git地址](https://github.com/baichuan-inc/Baichuan-7B.git)
>
> [Baichuan-7B-Chat模型下载](https://huggingface.co/baichuan-inc/Baichuan-7B-Chat)

