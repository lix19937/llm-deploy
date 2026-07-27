

**建一个无人值守的实验环路，让 AI 在项目中自主试验、评估、迭代，最终进化出比人类手动调优更高效的目标输出**    

----------------------------------


## Autoresearch 是什么    

给 AI Agent 一个小型但真实的 LLM 训练环境，让它自主实验：
`修改代码 → 训练 N 分钟 → 查看 val_bpb → 决定保留或 revert → 重复。`   
早上醒来，就能看到整夜的实验记录。

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py    # 下载数据、训练 tokenizer（一次性）
uv run train.py      # 每次实验固定 5 分钟
```

## 一、设计哲学      
传统的 AutoML（如 Optuna）是在人类定义好的超参数空间里进行数学搜索，而 autoresearch 则是将代码本身作为搜索空间。它的底层逻辑非常简单：

`约束目标 + 自动化可客观量化的指标评估 + 持续迭代机制` = 螺旋上升的收益      

为了能够稳定可靠地闭环完成这个任务，有3点限制：      
极度简化的状态管理（Git 即存储）：不用复杂的数据库，利用 Git 原生的 commit 和 rollback 来做实验日志和版本回滚。      
严格控制的单点修改：AI 被死死限制只能修改一个文件，降低复杂度。   
纯定量指标（Mechanical Verification）：拒绝任何主观的“看起来不错”，只看 Validation Loss 等硬性指标是否降低。     

```
       ┌────────────────────────┐
       │       program.md       │ <── (人类编写/修改基础指令与目标)
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             AI Agent                 │ ◄───────┐
│  (eg Claude Code / GPT-4/OpenCode)   │         │
└──────────────────┬───────────────────┘         │
                   │ (edit, save)                │
                   ▼                             │ 4. 评估/决策
         ┌───────────────────┐                   │ (根据指标 Keep 或 Rollback)
         │     train.py      │ ───┐              │
         └───────────────────┘    │              │
                                  │ 2. tran      │
                                  ▼              │
         ┌───────────────────┐ ───────> 验证指标 (如 Loss)
         │    prepare.py     │ (提供数据与评估工具)
         └───────────────────┘ 3. 5分钟快速验证
```

## 二、3层设计原则    

autoresearch 的核心框架。3个层次缺一不可，单独存在都没有意义。

### 原则 1：环境固定，搜索空间精确限定，变量明确     
```
prepare.py（固定不变）
  ├── 下载数据（HuggingFace，约400B tokens）
  ├── 训练 BPE tokenizer（vocab_size=8192）
  └── 定义共享常量（TIME_BUDGET=300, MAX_SEQ_LEN=2048）

train.py（Agent 唯一可修改文件，降低复杂度）
  ├── GPT 模型（自定义配置）
  ├── MuonAdamW 优化器
  └── 训练循环
```

如果 Agent 可以改评估框架，它可以通过调整指标让结果变好看。如果可以改数据，每次实验就不在同一个 ground truth 上比较。  
prepare.py 固定为"只读的环境参数"，   
train.py   是"Agent 的实验田"——这个分离确保了实验可比性。
禁止AI 拥有改动评估指标（Metric）或数据切分的权限。避免AI 最终会学会通过“作弊（Cheat）”（如过拟合验证集、泄露标签）来完成任务。

### 原则 2：固定时间预算，硬件比较    

每次实验固定 5 分钟（wall clock）。README 原文说这是 **"soft limit"**，目的是让不同 GPU 上的实验大致可比——H100 能跑更多步但吞吐量高，A100 能跑的步数更少但吞吐量低，但 val_bpb 只和模型压缩效率相关，不同硬件之间可以近似比较。

### 原则 3：单一指标驱动决策    

val_bpb（验证集每字节比特数，越低越好）是唯一的优化目标。这个指标的选择很讲究：   

与词表大小无关：不同 vocab_size 的模型可以直接比较    
与序列长度无关：不受上下文窗口大小影响   
物理含义清晰：衡量模型对字节序列的压缩效率     

## 三、代码核心设计  

### train.py：模型架构与优化器    
这是整个架构的核心受体。它是一个完整的、包含 GPT 架构、优化器（如 Muon、AdamW）和训练循环的单文件代码。   
全部放开：AI 可以任意修改这里的网络结构、学习率、优化器逻辑、Batch Size 甚至加入全新的正则化思路。   
单点突变：由于所有逻辑都在一个文件里，AI 生成的 Diff 极度清晰，不易产生跨文件解耦导致的逻辑混乱。     

### 滑动窗口注意力

大多数 Transformer 的注意力是"全局"的——每个 token 都能看到所有其他 token。在长序列上这是 O(n²) 的计算量。autoresearch 用了滑动窗口注意力：大部分层只能看到 1024 个 token（半上下文），只有最后一层能看到完整的 2048 个 token。

这样计算量大幅降低，同时保留了一层全局注意力确保模型"能看到"完整序列。

### Value Embeddings（最独特的设计）   

传统 Transformer 的 value 是由输入 embedding 计算的。autoresearch 在部分层额外加入了一个可学习的 value embedding，通过门控机制动态注入：
```
gate = 2 * sigmoid(ve_gate(x))  # 初始化为 1.0（中性注入）
v = v + gate * ve               # 可学习地控制注入程度
```   
这让模型在训练过程中自动学会：哪些层应该启用这个额外的 value，哪些层不需要。
传统的 Transformer 中，Value 向量（$V$）是输入 Token 经过线性变换 $W_v$ 动态计算出来的（即 $\text{Context-dependent}$）。而 Karpathy 引入的可学习 ve（Value Embedding）是一个与上下文无关、全局共享的静态参数。这就像是给大模型装上了一个“全局备忘录”或“通用先验知识库”。门控机制（gate）让模型自己决定：哪些层需要依赖当前上下文去推导 Value，哪些层可以直接查阅这个“全局备忘录”。这极大地释放了网络的表征能力。


### MuonAdamW：专门为 Transformer 设计的优化器   

这是 train.py 最复杂的一部分。标准优化器（AdamW）更新参数时，权重矩阵的奇异值会逐渐失衡，导致训练不稳定。MuonAdamW 在每次参数更新后，对梯度做一次正交化处理——让梯度矩阵的列（或行）互相正交，改善梯度流动。  

对于 Transformer 这种深层网络来说，这个优化能显著提升训练效率。

为什么 5 分钟的短时训练能有参考价值————因为 Muon 配合滑动窗口注意力在训练前期的收敛速度极快、斜率极陡，短时间就能分出架构的优劣。

### prepare.py：数据管道   

提供数据准备、BPE 分词器训练以及不可篡改的 Dataloader 和 Evaluation（评估逻辑）。     
防止 AI “作弊”：如果允许 AI 修改测试集或评估逻辑，它可能会通过“泄露标签”或“降低评估标准”来伪造高分。因此，prepare.py 作为规则裁判，是对 AI 绝对只读的。

### 最佳适应打包（Best-fit Packing）    

大多数 DataLoader 用"固定 batch + padding"：每个序列都 padding 到固定长度，浪费大量计算在 padding token 上。

autoresearch 用最佳适应算法：把文档序列紧凑地拼在一起，刚好填满 2048 token 的序列长度，一个 padding token 都没有。

这个设计让 GPU 计算100%用于真实数据，没有浪费。

## 四、program.md 的提示词工程设计     

这是给 AI Agent 的任务书。里面定义了当前的研究方向、任务背景和明确的优化指标。AI 会仔细阅读该文件以理解它需要做什么。    

### 4.1 核心挑战：对抗 Agent 的三大失败模式     

Karpathy 写这段提示词时，脑子里想的是："如果让 Agent 通宵跑，它会在哪里卡住？"

他预判了三个失败模式，针对性地埋了反制指令：   

|失败模式	            |表现	               |program.md 的反制|     
|-------------------- |----------------    | -------------- |      
|频繁停下来问"继续吗"	 |每次遇到选择就停止     |NEVER STOP 指令            |     
|陷入死循环无法推进	    |同一方向反复尝试	    |LOOP FOREVER + git 分支推进|     
|方向走偏而不自知	       |只看指标，忽略代码质量 |Simplicity criterion 框架|   

### 4.2 黑名单 + 白名单：消除歧义    
```
**What you CAN do:**
- Modify `train.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify `prepare.py`.
- Install new packages or dependencies.
- Modify the evaluation harness.
```   
不是"尽量不要改"而是"绝对不能改"。边界越清晰，Agent 越不需要停下来问。

### 4.3 单一目标 + 消除顾虑   
```
The goal is simple: get the lowest val_bpb.
Everything is fair game.
```   
先给一个简单到可以写在一句话里的目标，再告诉 Agent"什么都可以试"。这个顺序很重要——如果先列一堆约束，Agent 会倾向于保守；如果先说目标，再说边界，Agent 会更敢放手探索。

### 4.4 Simplicity criterion：防止优化陷阱    
这是整个提示词最精彩的一段：
```
All else being equal, simpler is better.
A small improvement that adds ugly complexity is not worth it.
Conversely, removing something and getting equal or better results
is a great outcome — that's a simplification win.
```   
三个具体数字判断案例：
```   
0.001 val_bpb 提升 + 20 行 hack 代码 → 不值得   
0.001 val_bpb 提升 + 删除代码 → 值得保留    
几乎无提升 + 代码简化 → 保留   
```
把"代码简洁性"本身定义为一种胜利，防止 Agent 陷入"指标提升但代码越来越烂"的优化陷阱。

### 4.5 最高指令：LOOP FOREVER + NEVER STOP   
```
LOOP FOREVER:

NEVER STOP: Once the experiment loop has begun...
do NOT pause to ask the human if you should continue.
```   
这是整个提示词的核心约束。"might be asleep"这个场景描写不是讲道理，而是构造一个具体画面：如果你停下来，人类会不高兴。比"你应该自主工作"这种抽象指令有效得多。

### 4.6 困境突围的行为菜单   
```
If you run out of ideas, think harder —
read papers referenced in the code,
re-read the in-scope files for new angles,
try combining previous near-misses,
try more radical architectural changes.
```   
当 Agent 说"我没想法了"，给它四个具体的行动菜单，而不是一句空洞的"再想想"。这是把"思考"这个模糊动作变成可执行的行为清单。

### 4.7 整体结构：五步控制流   
```
Setup → 自动化初始化，减少人工介入
   ↓
Experimentation → 明确权限，定义评估框架
   ↓
Output format → 标准化输出，减少歧义
   ↓
Logging → 强制结构化，防呆设计
   ↓
The experiment loop → 最高指令 + 具体控制逻辑
``` 

每一步都在解决 Agent 执行长任务时的具体问题，不是理论上的完美提示词模板，而是针对已知失败模式的事先反制手册。

### 4.8 为什么是轻量级 skill 模板     
Karpathy 称 program.md 为"super lightweight 'skill'"—— 这和 OpenClaw 的 SKILL.md 本质相同：  
给 Agent 提供特定领域的指令模板，让它在给定范围内自主行动。

区别在于粒度：OpenClaw skill 定义的是工具能力，program.md 定义的是研究流程。但核心逻辑一样：边界清晰 + 目标单一 + 循环可控。

## 五、实战：在 Linux 上运行自己的实验  

### 5.1 具体步骤
tbd   


### 5.2 Claude CLI 能否真正"一直运行下去"？ 

这是最实际的挑战。大多数 AI 产品被设计为"随时响应人类中断"，而 program.md 要求"用户睡觉你也继续跑"——这与产品设计哲学存在张力。    

claude 命令直接进入交互模式，默认允许读写。但 Agent 在遇到错误时仍可能主动停下来问"怎么办"。    

一个现实的预期：Claude Code 能运行较长时间（数小时），但不一定能真正通宵无人值守。

### 5.3 "AI 自己读论文"是真的吗？    
program.md 第 112 行提到：read papers referenced in the code。这是对 Agent 自身能力的调用，不是项目内置功能。

Claude Web/App 内置网页搜索，可以直接搜 arXiv 论文。当 Agent 思路枯竭时，它可以说"让我搜一下相关论文"，自己读摘要、提取 idea，再落实到 train.py 的改动上。

## 六、局限性   
### 6.1 任务范围极窄    
只能优化"给定训练代码和数据下的配置"。无法自主发现新任务、提出新假设、设计新实验范式。

### 6.2 AI 没有真正的科学理解    
Agent 发现某个优化有效，但不理解为什么有效。在工程上 OK，在科学上远远不够。

### 6.3 多 Agent 协作的可能性    
Karpathy 提到的下一个方向：多 Agent 异步大规模协作——一个 Agent 负责提出假设，一个负责实现代码，一个负责分析结果，一个负责文献调研。它们异步通信，模拟真实学术团队的工作模式。这会比单一 Agent 强多少？目前没有答案。

### 6.4 容易陷入局部最优（Local Search Trap）  
AI 经常会倾向于在“上一次成功的变体”附近做缝缝补补的微调（比如不断微调学习率），而缺乏跳出框架、尝试激进新架构的“冒险精神”（部分原因是 RLHF 让模型偏向于保守和安全）。   

### 6.5 算力依赖与种子黑客（Seed Hacking）   
由于只跑 5 分钟，AI 有时会敏锐地发现某种能够让“短期 Loss 骤降、但长期可能过拟合”的代码 trick。   

## 七、启示  

### 7.1 反馈闭环   
autoresearch 的成功不是因为"AI 很聪明"，而是因为设计者**精确地限定了搜索空间、定义了可评估的目标、并建立了可靠的反馈循环**。赢在用最简单的机制建立了极简反馈闭环。证明了在 Agent 时代，复杂的框架堆砌容易带来灾难，而基于 “Git 状态机 + 定量指标评判 + 单文件可变” 的极简主义架构，才是实现 AI 自我演进的最稳定解。

对于 AI 研究者和工程师的启示：与其争论 AI 能不能做研究，不如把精力放在**设计更好的"研究环境"上。当评估指标清晰、搜索空间精确、反馈及时**，AI 的自主研究能力会超乎预期。

### 7.2 从“软件工程”升级到“AI 时代的研发范式”   

Software 1.0：人类写死逻辑（C++/Python 代码）。    
Software 2.0：人类写网络架构和数据流，AI 训练权重（神经网络参数）。   
Software 3.0（Autoresearch 范式）：人类不写架构了，人类只写“环境、裁判（Loss）和边界（Sandbox）”，AI 自动在沙盒里通过生成 Software 2.0 的代码来逼近最优解。 如何通过确定性的工程结构，去约束和驱动不确定的 LLM 表现。



让AI 分析 autoresearch 项目   

https://github.com/davebcn87/pi-autoresearch   
