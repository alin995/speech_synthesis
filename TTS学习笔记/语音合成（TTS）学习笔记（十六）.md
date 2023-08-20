# 语音合成学习（十五）学习笔记

---

# 如何利用sadtalker生成ai alin

## 🔥 Highlight

- 🔥 The extension of the [stable-diffusion-webui] 

- 🔥 `SADTalker` 

|      |                             合成                             |                input image                 |
| :--: | :----------------------------------------------------------: | :----------------------------------------: |
|      | <video  src="https://github.com/alin995/speech_synthesis/assets/74090594/c33a9dcd-46a5-4fb4-9d7c-d8092665f4f5" type="video/mp4"> </video> | <img src='/img/Wwl.png' width='380'> |



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
3. 点击 [GFPGANv1.4](https://link.csdn.net/?target=https%3A%2F%2Fgithub.com%2FTencentARC%2FGFPGAN%2Freleases%2Fdownload%2Fv1.3.0%2FGFPGANv1.4.pth) 即可下载，将下载好的模型放到项目中

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

<img aligin="center" src="/img/searchhttp.png" />



二、中文插件
1.安装插件
stable diffusion项目的汉化插件就在Extensions扩展里。我们选择Extensions => availabel => 把hide Extensions with tags下面的几个功能按键取消 => search 直接搜索zh_CN localization 找到zh_CN localization插件，直接点击右边的install安装即可 <img aligin="center" src="/img/loadchinese.png" />

点击 extensions >> installed,查看我们是否成功的安装了插件，选择我们刚刚下载的的stable-diffusion-webui-localization-zh_CN汉化插件，点击apply and restart UI,这里我们就成功安装了汉化插件。

<img aligin="center" src="/img/anzhuang.png" />

2.设置插件
我们需要选择setting>>user interface>>localization，选择zh_CN

点击apply settings即可，这里由于汉化设置需要重启UI界面，我们可以直接点击reload UI界面

<img aligin="center" src="/img/setting.png" />

重启后，汉化完成了！！！

通过Webui实现生成
<img aligin="center" src="/img/webui.png" />

<img aligin="center" src="/img/webui.jpg" />
启动界面可以大致分为4个区域【模型】【功能】【参数】【出图】四个区域

1. 模型区域：模型区域用于切换我们需要的模型，模型下载后放置相对路径为/modes/Stable-diffusion目录里面，ckpt、pt等模型文件请放置到上面的路径，模型区域的刷新箭头刷新后可以进行选择。
2. 功能区域：功能区域主要用于我们切换使用对应的功能和我们安装完对应的插件后重新加载UI界面后将添加对应插件的快捷入口在功能区域，功能区常见的功能描述如下

- txt2img（文生图） --- 标准的文字生成图像；
- img2img （图生图）--- 根据图像成文范本、结合文字生成图像；
- Extras （更多）--- 优化(清晰、扩展)图像；
- PNG Info --- 图像基本信息
- Checkpoint Merger --- 模型合并
- Textual inversion --- 训练模型对于某种图像风格
- SadTalker --- 生成会说话的图片
- Settings --- 默认参数修改
- Extensions --- 扩展

3.参数区域：根据您选择的功能模块不同，可能需要调整的参数设置也不一样。例如，在文生图模块您可以指定要使用的迭代次数，掩膜概率和图像尺寸等参数配置

4.出图区域：出图区域是我们看到AI绘图的最终结果，在这个区域我们可以看到绘图使用的相关参数等信息。

---

## 相关资源

> [SadTalker git地址](https://github.com/Winfredy/SadTalker) 
>
> [SadTalker官网](https://github.com/OpenTalker/SadTalker/blob/main/docs/webui_extension.md)

