## Prompt 是什么？  
先略过。

## Token 是什么？
+ 是分词器（tokenization）把文本（语言模型，如果是多模态的，如图像patch变换后的向量）分出来的"最小单元"。    
+ 离散的符号。

考虑句子：
```
"The quick brown fox jumps over the lazy dog." 假设分词器认为这个句子中的每个单词就是一个token。如果我们将这个句子分解成单词级别的token，那么包含的token有：
"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"。这样，在处理这个句子时，每个单词就是一个token。
```

## Token ID 是什么？
+ 在模型里，它只是一个整数索引。

比如：
```
"The" → Token ID: 1342
"quick" → Token ID: 2567   
"brown" → Token ID: 567   

```
所以，Token **本身没"向量值"，它只是个词典里的编号**。

## Embedding 是什么？
+ Embedding 是token id 的向量化表示 （将token id转换成高维向量）。
+ 每个 Token 会在词表（Vocab）里有一个固定的向量表示。
+ 这个向量在模型训练时一起学习出来（捕获语义和上下文关系）。

```
Input: "DeepSeek is amazing!"
分词器 → ["Deep", "Seek", " is", " amazing", "!"]
Token ID → [5231, 11987, 456, 7812, 34]
```
```
Input: "深度寻迹很好用！"
分词器 → ["深", "度", "寻", "迹", "很", "好", "用", "！"]
Token ID → [341, 487, 292, 653, 512, 876, 90, 12]
```

比如：
```
Token ID: 1342 → Embedding: [0.12, -0.07, 0.58, ...]
Token ID: 2567 → Embedding: [0.44, 0.01, -0.32, ...]
```

所以：  
```
Embedding：Token ID --> R^d
```   

这里 d 是 Embedding 维度（比如 768, 1024, 4096）。

## Token → Embedding 是怎么做的？
在模型里：
有一个叫 Embedding Lookup Table 的权重矩阵：`R^Vxd`    
其中 V 是词表大小（比如 50k），d 是向量维度。    
对于每个 Token：
    用 Token ID 作为索引，取出第 i 行，就是它的 Embedding。
这一步不需要计算，就是查表：`Embedding = E(token id) --> R^d`


## 视觉 Token 和 Embedding 是一样的思路吗？
视觉 Token 和文本 Token 最大的区别：  
+ 文本 Token 是离散符号 → 查表。
+ 视觉 Token 是图像 patch 经过线性变换直接生成向量。   

比如：    
ViT：把图像切成 Patch（比如 16x16）
每个 Patch 展平（flatten）成向量，然后过一个 Linear Layer：`patch embedding = W* patch_pixels + b`    
所以：**视觉 Token → 没有离散 ID，是直接把局部像素映射成向量**。

## Embedding 和 Token 的关系总结

 
|    | 文本	|图像 |     
|----|-----|-----|   
|Token 是什么	|离散的 ID |	切块的 Patch|    
|Embedding 怎么来 |	查词向量表 |	Patch 做线性变换 |  
|Embedding 用来干嘛 |	输入到 Transformer|	输入到 Transformer |   

最后都变成：
`Token Embedding Sequence = [E1, E2, E3, ..., EN] R^Nxd`

## 位置编码（Positional Embedding）
还有个要点：
+ Token 本身没顺序感知能力。
+ 所以 Embedding 还要加上位置编码：`Xinput = Etoken + Eposition`    

不管是文字还是图片，都是：   
+ 文本 Token：加 1D 位置编码。
+ 图像 Patch：加 2D 位置编码（XY 位置）。  

## 公式串起来就是：
【文本】  
`Token ID --LookUp --> Token Embedding + 1D Positional Embedding --Transformer--> Output`

【视觉】   
`Patch Pixels --Linear --> Patch Embedding + 2D Positional Embedding --Transformer--> Output`

## 总结   
> Token 是离散 ID，Embedding 是把 ID 变成向量，才能被 Transformer 处理；文本靠查表，图像靠 Patch 投影，一起丢进同一个多模态自注意力里！


### 什么是 Prompt？   
Prompt = 给大模型的输入指令 / 上下文提示。
可以是一段对话、系统设定、示例、图片描述、角色扮演等。
本质就是一段 Token 序列，用 Token ID 表示，然后进入 Transformer    

```
用户输入：写一首关于春天的诗。
完整 Prompt：
[系统消息] 你是一个中文诗人。
[用户消息] 写一首关于春天的诗。

分词 → Token → Embedding → Transformer
```

### Prompt 和 Token 的关系 一句话总结
+ Prompt = Token 的语义组合
+ Token = Prompt 的底层编码单元

训练和推理时都是：
```
Prompt (文本)  → 分词器 → Token ID → Embedding → 模型生成下一个 Token
```
