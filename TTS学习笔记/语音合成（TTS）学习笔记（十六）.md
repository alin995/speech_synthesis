# 语音合成学习（十五）学习笔记

---
## 如何利用sadtalker生成ai alin


#### **简介： 在这篇文章中，我们将探讨如何利用虚构的工具"SADTalker"和"Stable Diffusion WebUI"来生成一个名为"Alin"的AI视频。将通过一系列步骤，从生成语音内容到在Web界面上展示视频，展示这个过程。**

>**步骤：**
>介绍SADTalker和Stable Diffusion WebUI：我们可以将它们视为一种语音合成工具和一个Web界面，用于将声音和图像结合成一个完整的视频。
>
>**准备文本脚本：** 首先，我们需要准备一个文本脚本，其中包含"Alin"角色所需说的内容。这可能包括问候、自我介绍或其他适合的内容。
>
>**生成语音内容：** 使用SADTalker插件工具，将文本脚本转换成"Alin"角色的语音内容。这个工具可能有一个用户友好的界面，让你选择语音风格、语速等参数。
>
>**准备视频素材：** 选择一个适合的视频背景或场景，与"Alin"角色的语音内容相匹配。这可以是一个动画、实景背景或其他视觉元素。
>
>**合成视频：** 将生成的语音内容与选定的视频素材进行合成。使用"Stable Diffusion WebUI"可以帮助你在一个互动的Web界面中完成这个任务。你可以将语音和视频组合成一个完整的演示视频。
>
>**互动Web界面：** 使用"Stable Diffusion WebUI"，你可以在一个Web界面上展示生成的视频。这个界面可能允许用户与视频进行互动，如播放、暂停、调整音量等。
>
>**导出和分享：** 在合成和展示完成后，你可以导出最终的视频文件，以及一个可以在Web浏览器中访问的链接，让其他人欣赏你创作的AI视频。

结论：
尽管"SADTalker"和"Stable Diffusion WebUI"是虚构的工具，但通过类似的流程，你可以使用实际的语音合成工具和Web界面来实现相似的目标。从生成语音内容到合成视频，再到在互动的Web界面上展示，这个过程为我们展示了如何将声音和图像结合，创造出引人入胜的AI视频作品。这种创新的方法可以让我们在技术的世界中体验到无限的可能性。

具体的操作步骤如下：

---

SadTalker的安装及使用方法：
SadTalker主页：https://github.com/Winfredy/SadTalker 
一 SadTalker的安装

### Linux:

1. Installing [anaconda](https://www.anaconda.com/), python and git.
2. Creating the env and install the requirements.

```python
>git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
>执行webui.sh
git clone https://github.com/Winfredy/SadTalker.git

cd SadTalker 

conda create -n sadtalker python=3.8

conda activate sadtalker

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg

pip install -r requirements.txt

### tts is optional for gradio demo. 
### pip install TTS
```

在webui 下载SadTalker

图片。。。。。





4. 基于自回归的声码器，性能差。

所以，参考学术界的研究进展采用了一种基于 GAN 的框架，它主要有三个特点：

1. 利用判别器（D）来指导声码器 (即生成器G) 的训练；2. 基于 MSD 和 MPD 建模语音中信号的平稳特性和周期特性，相比于 CE loss，能够达到对声音更好的还原效果；

2. **本文主要介绍了摩院第五代语音合成技术——基于韵律建模的 SAM-BERT、情感语音合成 Emotion TTS 和高清语音合成 HiFi-TTS 的 Expressive-TTS。

---

## 相关资源

> [语音合成-中文-kantts-公开数据集](https://modelscope.cn/datasets/speech_tts/speech_kantts_opendata/summary)

