# 语音合成学习（二）学习笔记

---

### 练习项目案例

* [百度paddlepaddle平台代码](https://github.com/PaddlePaddle)

  > 官网地址   https://www.paddlepaddle.org.cn/
  >
  > 语音地址   https://github.com/PaddlePaddle/PaddleSpeech
  >
  > 模型地址   https://www.paddlepaddle.org.cn/modelsDetail?modelId=26

* [ Tacotron-2项目代码 ](https://gitee.com/goohere/Tacotron-2)

  > 官网地址    https://www.anaconda.com/
  >
  > [Tacotron2讲解](https://blog.csdn.net/suiyueruge1314/article/details/107185195)
  >
  > [Tacotron2 模型详解](https://blog.csdn.net/qq_37236745/article/details/108846686)

---

## 项目运行笔记

**前提准备：**

> *此项目在服务器操作  项目在svn 。安装svn 需要brew ；也可下载可视化svn工具，例：cornerstone，SnailSVN 等*
>
> ---
>
> **安装svn**
>
> 先查看本机是否安装brew  ➡️     brew —v
>
> 没有的话需安装brew   [ **安装brew** ](/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)”)
>
> 安装完毕 ，重启中端 ➡️    source /Users/wangwenlin/.zprofile
>
> 安装 ➡️   svn brew install subversion  ｜      查看版本 ➡️    svn --version 
>
> ---
>
> 下载代码 ➡️    <font>svn checkout xxxxxxxxx</font>
>
> ⚠️第一次连接ssh 时候，会产生一个本地证书在本机.ssh目录下  提示选择Y信任即可
>
> ---
>
> 安装python环境 （mac 默认是python2）需下载python3
>
> [ **python3.10下载** ](https://www.python.org/downloads/macos/)
>
> 命令行方式⬇️
>
> <p>sudo apt install python 3.9</p>
>
> <p>python  --version</p>
> <img align=“center” src="/img/pythonversion.jpg" />
>
> [ **安装anaconda** ](https://www.anaconda.com/products/distribution)
>
> 命令行方式⬇️
>
> cd /services/current_apps/build
>
> sudo wget -r -c -nH https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
>
> sudo bash Anaconda3-2022.05-Linux-x86_64.sh
> source /services/current_apps/anaconda3/bin/activate
>
> ---
>
> **连接ssh**  ➡️      sudo  -i    |       ssh xxxx@10.xx.xxx.xx 
>
> ---
>
> **创建python沙盒⬇️ **
>
> conda create -n urname python=3.6 tensorflow=1.10
> conda activate urname
>
> 激活环境  conda activate urname    命令行前缀改变 urname =》 bingo！
>
> ----
>
> **更换镜像源！！！**
>
> pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
>
> pip3 config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
>
> <table>
>     <tr><td>阿里云</td><td>https://mirrors.aliyun.com/pypi/simple/</td></tr>
>     <tr><td>中科大</td><td>https://pypi.mirrors.ustc.edu.cn/simple/</td></tr>
>     <tr><td>清华</td><td>https://pypi.tuna.tsinghua.edu.cn/simple/</td></tr>
>     <tr><td>豆瓣</td><td>http://pypi.douban.com/simple/</td></tr>
> </table>

---

## 百度paddlepaddle项目 ##

### 1. 简介

PP-TTS 是 PaddleSpeech 自研的流式语音合成系统。在实现[前沿算法](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#text-to-speech-models)的基础上，使用了更快的推理引擎，实现了流式语音合成技术，使其满足商业语音交互场景的需求。

#### PP-TTS

语音合成基本流程如下图所示：
<img src="/img/tts.png"/>

PP-TTS 默认提供基于 FastSpeech2 声学模型和 HiFiGAN 声码器的中文流式语音合成系统：

- 文本前端：采用基于规则的中文文本前端系统，对文本正则、多音字、变调等中文文本场景进行了优化。
- 声学模型：对 FastSpeech2 模型的 Decoder 进行改进，使其可以流式合成
- 声码器：支持对 GAN Vocoder 的流式合成
- 推理引擎：使用 ONNXRuntime 推理引擎优化模型推理性能，使得语音合成系统在低压 CPU 上也能达到 RTF<1，满足流式合成的要求

###  2. 语音合成任务 ###

##### 数据集：
常见语音合成数据集如下表所示：

<img align=“center” src="/img/shujvji.jpg"/>

### 3. 模型如何使用

### 3.1  配置项目环境   ###

###        安装 paddlespeech

​        此项目需求➡️   gcc >= 4.8.5  paddlepaddle >= 2.4.1 python >= 3.7  创建对应版本沙盒环境 

```javascript
sudo apt install gcc
sudo apt install g++
sudo apt install gcc g++ libopenblas-dev liblapack-dev libatlas-base-dev libblas-dev
pip install paddlespeech
```

```javascript
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```

添加执行文件 textway.py

```javascript
from  paddlespeech.cli.tts **import** TTSExecutor
tts_executor = TTSExecutor()
wav_file = tts_executor(
 text="热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！",
output='output.wav',
am='fastspeech2_mix',
voc='hifigan_csmsc',
lang='mix',
spk_id=174)
```
<img align=“center” src="/img/outputwav.png"/>

### 4. 模型原理

### 4.1 声学模型 FastSpeech2
<img align=“center” src="/img/fastSpeech.png"/>


PaddleSpeech TTS 实现的 FastSpeech2 与论文不同的地方在于，我们使用的的是 phone 级别的 `pitch` 和 `energy`(与 FastPitch 类似)，这样的合成结果可以更加**稳定**。
<img align=“center” src="/img/fastPitch.png"/>

###  4.2 声码器 HiFiGAN

1. 引入了多周期判别器（Multi-Period Discriminator，MPD）。HiFiGAN 同时拥有多尺度判别器（Multi-Scale Discriminator，MSD）和多周期判别器，目标就是尽可能增强 GAN 判别器甄别合成或真实音频的能力。

2. 生成器中提出了多感受野融合模块。WaveNet为了增大感受野，叠加带洞卷积，逐样本点生成，音质确实很好，但是也使得模型较大，推理速度较慢。HiFiGAN 则提出了一种残差结构，交替使用带洞卷积和普通卷积增大感受野，保证合成音质的同时，提高推理速度。
<img align=“center” src="/img/mpd.png"/>

<font align="center" size=4 color="violet">出现问题的排查</font>

⚠️  pip install paddlespeech 时需要先激活沙盒环境！！！！在当前对应版本环境装否则会出现（因为版本不匹配）安装很久装不上的问题

⚠️添加执行文件时 需要先把本地文件上传服务器

---

## Tacotron-2项目 ##

> ubuntu服务器 和 GPU环境centos 环境分别运行的流程

---

## ==ubuntu服务器== 

### 1. 配置项目环境  ###

有些项目python 和 tensorflow 版本没有指定对 需要自己根据去推断版本好

拉取项目 https://gitee.com/goohere/Tacotron-2

创建对应版本的python沙盒环境

> conda create -n tacotron python=3.6 tensorflow=1.9
> conda activate tacotron

### 1.1安装编译器及依赖包

```
sudo apt install gcc
sudo apt install g++
sudo apt install gcc g++ libopenblas-dev liblapack-dev libatlas-base-dev libblas-dev
```

接下来，安装一些Linux依赖项以确保音频库正常工作：

> apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools

<font align="center" size=4 color="violet">出现问题的排查 </font>
<img align=“center” src="/img/installerror.png"/>

​           **安装环境依赖时，一个一个单独装！！**

**|**    如果出现! <img align=“center” src="/img/install02.png"/>

<img align=“center” src="/img/fix02.png"/>

最后，您可以安装 requirements.txt. 如果你是一个 Anaconda 用户: (可以用 **pip3** 代替 **pip** 并 用**python3** 代替 **python**)

> pip install -r requirements.txt

**|**    安装依赖包时，需要注意；如果提示路径不对 ，建议一个一个安装指定版本的包

```
pip3 install \
    falcon==1.2.0 \
    inflect==0.2.5 \
    audioread==2.1.5 \
    librosa==0.5.1 \
    matplotlib==2.0.2 \
    numpy==1.14.0 \
    scipy==1.0.0 \
    tqdm==4.11.2 \
    Unidecode==0.4.20 \
    sounddevice==0.3.10 \
    lws \
    keras

sudo apt install python3-pyaudio

pip3 install llvmlite==0.29.0 numpy==1.14.0 numba==0.45.1 pydub==0.23.1
pip3 install keras==2.2.4
```

⚠️ 安装前提激活python环境

**|**    安装完毕 检查一下 👉    pip  list

---

### 2. 运行步骤

- 步骤 (0): 获取数据集, 这里我设置了Ljspeech的示例。
- 步骤 (1): 预处理数据。
- 步骤 (2): 训练你的Tacotron模型。产生logs-Tacotron文件夹。
- 步骤 (3): 合成/评估Tacotron模型。给出tacotron_output文件夹。
- 步骤 (4): 训练您的Wavenet模型。产生logs-Wavenet文件夹。
- 步骤 (5): 使用Wavenet模型合成音频。给出wavenet_output文件夹。

**注意：**

- 步骤2,3和4可以通过Tacotron和WaveNet（Tacotron-2，步骤（*））的简单运行来完成。
- 原有github的预处理仅支持Ljspeech和类似Ljspeech的数据集（M-AILABS语音数据）！如果以不同的方式存储数据集，则需要制作自己的preprocessing脚本。
- 如果同时对两个模型进行训练，则模型参数结构将不同。

### 3.模型架构
<img align=“center” src="/img/tacotron2.png"/>

### 4. 预处理 ###

执行以下步骤之前，请确保您在项目文件夹中

```javascript
cd Tacotron-2
```

然后可以使用以下命令开始预处理：

```javascript
python preprocess.py
```

以使用 **–dataset** 参数选择数据集。如果使用**M-AILABS**数据集，则需要提供 **language, voice, reader, merge_books and book arguments** 以满足您的自定义需求。默认是 **Ljspeech**.

⚠️  前提下在好数据集包放在项目文件夹在执行预处理

### 5. 训练

按顺序训练两个模型:

征预测模型**Tacotron-2**可以分别被训练使用：

```javascript
python train.py --model='Tacotron-2'
```

每5000步记录一次，并存储在**logs-Tacotron**文件夹下。

<img align=“center” src="/img/t2.png" />

<img align=“center” src="/img/0101.png" />

⚠️ 如果出现以下报错<img src="/img/t2erroe.png" />


==此时注意查看 tensorflow版本， 1.9限定了keras的版本范围==

当然，单独训练**wavenet**是通过以下方式完成的：

```javascript
python train.py --model='WaveNet'
```

logs will be stored inside **logs-Wavenet**.

**注意：**

- 如果未提供模型参数，则训练将默认为**Tacotron-2**模型培训。（与tacotron模型结构不同）
- 训练模型的参数可以参考 [train.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/train.py) 有很多选项可以选
- wavenet 的预处理可能得单独使用 **wavenet_proprocess.py**脚本

################################################################

## ==GPU环境centos==

**gpu环境下需要安装驱动cuda，CUDNN。cudnn -> 基于cuda驱动的库。提前下载好驱动包**

**查看本机信息选择合适版本的cuda/cudnn 版本**（下载地址：相关阅读）
<img align=“center” src="/img/installcuda.png" />


### 1. 配置项目环境   ###

> 创建对应版本python环境

​             conda create -n tacotron python=3.6 tensorflow=1.9 （*<u>此操作后续和在unbantu服务器上操作的一样，操作命令符不同。有些依赖包的包名有些不同</u>*）

> 创建gpu对应版本的环境

​              conda create -n tacotron-gpu python=3.6 tensorflow-gpu=1.9

<font align="center" size=4 color="yellow">  配置环境出现的问题：</font>

**｜**安装驱动包时，提前确定哪个是基础包哪个是补丁包，安装时需注意文件名

如：<img align=“center” src="/img/budingyuan.png"/>


解决一：修改/添加 yum源

解决二： 查看对应centos 环境的包名 👇（参考相关阅读）

｜yum -y install portaudio portaudio-devel

｜ pip install pyaudio

如：多个yum运行<img align=“center” src="/img/yumerror.png"/>

解决：  kill 对应的pid

### 2. 运行步骤

同ubuntu服务器运行操作一样     python train.py --model='Tacotron-2'

预处理：
<img align=“center” src="/img/centoscpu.png"/>


cpu执行完成：
<img align=“center” src="/img/centoscpulook.png"/>


gpu执行完成：
<img align=“center” src="/img/gputimes.png"/>

<img align=“center” src="/img/gpu0101.png"/>

<img align=“center” src="/img/gpustep.png"/>


---

以上主要是对语音合成开源项目的练习过程做个整理总结，也作为自己的踩坑记录。希望能帮助到大家。

---

## 其他版本的Tacotron2开源项目

> - https://github.com/Rayhane-mamah/Tacotron-2
> - https://github.com/NVIDIA/tacotron2

## 相关阅读

> - cuda官网  https://developer.nvidia.com/
>
> - cudnn下载地址  https://developer.nvidia.com/cudnn 
>
> - cuda安装参考  https://blog.csdn.net/shiner_chen/article/details/125857553
>
> - centos环境下安装依赖包问题参考 https://blog.csdn.net/qq_34638161/article/details/80383914   
>
>   
