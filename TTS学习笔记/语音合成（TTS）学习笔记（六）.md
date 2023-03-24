# 语音合成学习（六）学习笔记

---

# 文本归一化

## 一、介绍

### 简介

>  文本正则化是将文本数据转化为标准形式的过程，以便于进行自然语言处理和文本分析。文本归一化通常包括以下步骤：
>
>  - 文本清理：删除文本中的无用信息，如HTML标记，特殊字符，URL和标点符号等。
>  - 大小写统一：将文本中的所有字符转换为小写或大写字母。
>  - 停用词过滤：删除指定的停用词，如“的”、“在”、“是”等常用词语。
>  - 词干提取或词形还原：将不同形式的词汇转换为其原始形式，如将“running”转换为“run”或将“cats”转换为“cat”。
>  - 拼写纠正：修正文本中的拼写错误。
>
>  文本归一化的目的是消除文本中的噪声和不规则性，提高自然语言处理和文本分析的准确性和效率。

------

**TN全称Text Normalization，意思是文本规整、文本正则化 。**

TN是 TTS (Text-to-speech，文本转语音) 系统中的重要组成部分，主要功能是将文本中的数字、符号、缩写等转换成语言文字。如：

```bash
# delete english characters
# e.g. "你好aBC" -> "你 好"

# fractionation
# e.g. "现场有7/12的观众投出了赞成票"  -> "现场有十二分之七的观众投出了赞成票"

# time
# e.g. "2:10pm出门" -> "下午两点十分出门"
```

- 使用归一化的好处：

> 1.提升模型的收敛速度(即加快梯度下降求最优解的速度
>
> 2.提升模型的精度
> 在涉及到一些距离计算的算法时效果显著，比如算法要计算欧氏距离，归一化可以让可以让各个特征对结果做出的贡献相同，未归一化就会造成精度的损失。

## 二、常见的归一化转换规则与表达 

```javascript
# 过滤掉特殊字符
text = re.sub(r'[——《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
# 日期表达式
用 / 或者 - 分隔的 YY/MM/DD 或者 YY-MM-DD 日期
RE_DATE2 = re.compile(r'(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])')
RE_DATE21 = re.compile(r'(\d{4}|\d{2})年'
                     r'((0?[1-9]|1[0-2])月)?'
                     r'(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?')
# 时刻表达式
RE_TIME = re.compile(r'([0-1]?[0-9]|2[0-3])'
                     r':([0-5][0-9])'
                     r'(:([0-5][0-9]))?')
# 时间范围，如8:30-12:30
RE_TIME_RANGE = re.compile(r'([0-1]?[0-9]|2[0-3])'
                           r':([0-5][0-9])'
                           r'(:([0-5][0-9]))?'
                           r'(~|-)'
                           r'([0-1]?[0-9]|2[0-3])'
                           r':([0-5][0-9])'
                           r'(:([0-5][0-9]))?')
# 分数表达式
RE_FRAC = re.compile(r'(-?)(\d+)/(\d+)')

# 百分数表达式
RE_PERCENTAGE = re.compile(r'(-?)(\d+(\.\d+)?)%')
# 整数表达式
# 带负号的整数 -10
RE_INTEGER = re.compile(r'(-)' r'(\d+)')
# 编号-无符号整形
# 00078
RE_DEFAULT_NUM = re.compile(r'\d{3}\d*')
# 数字表达式
# 纯小数
RE_DECIMAL_NUM = re.compile(r'(-?)((\d+)(\.\d+))' r'|(\.(\d+))')
# 正整数 + 量词
RE_POSITIVE_QUANTIFIERS = re.compile(r"(\d+)([多余几\+])?" + COM_QUANTIFIERS)
RE_NUMBER = re.compile(r'(-?)((\d+)(\.\d+)?)' r'|(\.(\d+))')
# 范围表达式
RE_RANGE = re.compile(r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))[-~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))')
# 规范化固话/手机号码
# 手机
# 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
# 联通：130、131、132、156、155、186、185、176
# 电信：133、153、189、180、181、177
RE_MOBILE_PHONE = re.compile(r"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)")
RE_TELEPHONE = re.compile(r"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{7,8})(?!\d)")

# 全国统一的号码400开头
RE_NATIONAL_UNIFORM_NUMBER = re.compile(r"(400)(-)?\d{3}(-)?\d{4}")

# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE = re.compile(r'(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)')
```

```
## Normalize unicode characters  // 规范化unicode 字符
def remove_weird_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore').decode(
        'utf-8', 'ignore')
    return text
    
## Remove extra linebreaks   //删除额外换行符
def remove_extra_linebreaks(text):
    lines = text.split(r'\n+')
    return '\n'.join(
        [re.sub(r'[\s]+', ' ', l).strip() for l in lines if len(l) != 0])
        
 
## Remove extra medial/trailing/leading spaces  // 删除额外的中间/后面/前面的空间
def remove_extra_spaces(text):
    return re.sub("\\s+", " ", text).strip()
    
## Seg the text into words   // 分词
def seg(text):
    text_seg = jieba.cut(text)
    out = ' '.join(text_seg)
    return out
    
## Remove punctuation/symbols  // 除去标点符号
def remove_symbols(text):
text = ''.join(
        ch for ch in text if unicodedata.category(ch)[0] not in ['P', 'S'])
    return text
    
## Remove numbers  // 除去数字
def remove_numbers(text):
    return re.sub('\\d+', "", text)
    
## Remove alphabets   //除去字母
def remove_alphabets(text):
    return re.sub('[a-zA-Z]+', '', text)
  
## Unify upper/lower cases   //转换大小写
    if args.to_upper:
        text = text.upper()
    if args.to_lower:
        text = text.lower()


```



以上主要是对归一化的简单介绍，也整理了一部分常用的规则。

---

## 相关阅读

- https://github.com/PaddlePaddle/PaddleSpeech/pull/658/files
- https://github.com/745165806/PaddleSpeechTask

