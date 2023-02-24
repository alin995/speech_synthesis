# 语音合成学习（三）学习笔记

---
# Gradio--搭建可视化演示环境

## 一、介绍
### 简介
>  Gradio定位是快速构建一个针对人工智能的python的webApp库，在Hugging Face等提供各种模型推理展示的平台广告使用，阿里的魔塔展示也是基于此。
>
> 大家思考下，Gradio作为一款python库，底层逻辑是什么？
>
> - 结果：Gradio展示的还是web元素
> - 过程：所以Gradio是即懂python又懂web开发（css/js/html)的开发者，通过python对这些web技术做了封装
> - pipline：python语言--> css/js/html

### 
> Gradio是MIT的开源项目。
>
> 使用gradio，只需在原有的代码中增加几行，就能自动化生成交互式web页面，并支持多种输入输出格式，比如图像分类中的图>>标签，超分辨率中的图>>图等。
>
> 同时还支持生成能外部网络访问的链接，能够迅速让你的朋友，同事体验你的算法。
>
> 总结起来，它的优势有：
>
> - 自动生成页面且**可交互**
> - 改动**几行代码**就能完成
> - 支持自定义多种输入输出
> - 支持生成**可外部访问的链接**进行分享
>
> 目前已经有很多优秀的开源项目使用Gradio做demo页面。那么该怎么使用Gradio，让我们一起来玩玩～

## 二 **Get start**

### **0.安装Gradio**

```bash
pip install gradio 或
为了更快安装，可以使用清华镜像源。
pip install gradio  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.**写个简单的**例子

```
# app.py
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch(server_name="0.0.0.0")

# 启动
# python web.py
# Running on local URL:  http://0.0.0.0:7860
# To create a public link, set `share=True` in `launch()`
```

![gradiowebUI](/Users/wangwenlin/Desktop/img/gradiowebUI.png)

上面的代码就是简单一个webApp，功能是输入一个文本，输出一个文本。代码中关键点：

- 导入包 `import gradio as gr`
- gr.Interface 构建一个app， 确定输入inputs和输出outputs的类型，已经处理输入inputs的函数（这个函数返回一个outputs的类型）
- 提供一个app的功能模块函数
- launch 启动一个web容器，对外提供服务

**梳理下web渲染流程**

- 根据输入输出类型（如text）封装html组件（with css样式，布局等）
- 点击submit：通过js获取输入的值传递（ajax）给后台处理函数（greet），通过js回调函数接收函数的返回值，然后通过js赋值给html元素

上面只是介绍了Gradio的简单的使用，Gradio提供了丰富的html组件，如文本框，图像，视频，下拉框，单选框等等。

**核心参数**

gradio的核心是它的`gr.Interface`函数，用来构建可视化界面。inputs和outputs都是可以多个，Gradio根据类型展示相应的组件

- fn：放你用来处理的函数
- inputs：写你的输入类型，这里输入的是图像，所以是"image"
- outputs：写你的输出类型，这里输出的是图像，所以是"image"

最后我们用`interface.lauch()`把页面一发布，一个本地静态交互页面就完成了！另外，可以通过`.launch(share=True)`来分享功能，这个功能可以生成一个域名，可以在外部直接访问。

```
interface.launch(inbrowser=True, inline=False, validate=False, share=True)
```

> `inbrowser` - 模型是否应在新的浏览器窗口中启动。 
>
> `inline` - 模型是否应该嵌入在交互式python环境中（如jupyter notebooks或colab notebooks）。 
>
> `validate` - gradio是否应该在启动之前尝试验证接口模型兼容性。 
>
> `share` - 是否应创建共享模型的公共链接。

### 2. 本人模拟了一个输入文本转换语音的例子

输入文本输出了指定的使用训练好的模型

![image-20230224143126097](/Users/wangwenlin/Desktop/img/t-v.png)

![image-20230224142018598](/Users/wangwenlin/Desktop/img/owngradio.png)

## 三  **gradio依赖包的版本** ##

#### 踩坑记录 ####

​	安装的gradio 是3.0版本  当前环境 tensorflow  1.9.0 ，执行时候报错 

![image-20230224151157510](/Users/wangwenlin/Desktop/img/error0.png)

原来是相互依赖，需要升级numpy的版本（当前训练模型合成语音的项目环境需要numpy ==1.14.0）。于是，查一下版本不匹配为了不影响训练合成功能，进行gradio降级 => 安装gradio2.0版本，搭建webUI界面显示了，重新执行了文件， 但是出现了语音无法合成，排查许久才发现，自动把numpy升级了。针对合成和执行实现web不能同时实现的问题：

<u>最终解决是通过多个包和版本一起指定</u>安装  完美解决 （gradio和numpy 有依赖关系）



以上主要是对如何使用gradio的简单介绍，也是用的较多的搭建可视化组件。

---
## 相关阅读
- [Grado官网](https://www.gradio.app/)
- github:https://github.com/gradio-app/gradio

