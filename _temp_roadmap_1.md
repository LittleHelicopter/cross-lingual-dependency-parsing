## 阶段一：基础设施构建 (Preparation)

*目标：将你的模型搬上“手术台”*

1. **环境配置**：
* 安装 `transformer_lens`。
* 将你微调好的 XLM-R 权重转换并加载到 `HookedTransformer` 类中（这是进行回路分析的前提）。


2. **构造对照数据集 (Clean/Corrupted Pairs)**：
* 回路分析的核心是对比。你需要准备：
* **Clean Prompt**：正常的源语言/目标语言句子。
* **Corrupted Prompt**：破坏了句法结构的句子（例如：随机打乱词序，或将关键动词替换为名词）。
* 例子：
| 编号 | Clean English (Ls) | Corrupted English (Ls_corr) | Clean Chinese (Lt) | Corrupted Chinese (Lt_corr) |
|------|--------------------|-----------------------------|--------------------|-----------------------------|
| 1 | I read the book. | the book I read | 我 读 了 这本 书。 | 书 这本 了 读 我。 |
| 2 | She likes apples. | likes she apples | 她 喜欢 苹果。 | 苹果 喜欢 她。 |
| ... | ... | ... | ... | ... |



3. **定义 Metric (评估指标)**：
* 回路分析需要一个标量输出。在你的任务中，可以是 **Logit Difference**（正确解析标签的 Logit 减去错误标签的平均 Logit）或 **Loss**。

---

## 阶段二：回路定位 (Circuit Discovery Pipeline)

*目标：找到那条“跨语言句法高速公路”*

### 0. 实验设计


#### 第一步：定位——寻找关键 Circuit

在进行“跨语言替换”之前，需要先知道在**单一语言**内，哪些 Head 是负责句法的。

1. **单语言 Patching (以英文为例)**：
* **输入 A**：`Clean English` (“I read the book.”)
* **输入 B**：`Corrupted English` (“book the read I.”)
* **操作**：逐个将 A 的 Head 激活值替换给 B。
* **结果**：比如第 6 层第 2 个 Head 和第 7 层第 5 个 Head 被替换后，B 的解析效果大幅提升。
* **结论**：这两个 Head 是英文的**关键句法回路 (English Syntactic Circuit)**。
* **实现**：transformerlens直接实现
* **寻找组合**：top-k 影响的head

#### 第二步：跨语言验证——神经元替换手术

验证英文的“知识”能否直接修补中文的“损坏”。

1. **准备环境**：
* 模型 A 处理 `Clean English`。
* 模型 B 处理 `Corrupted Chinese`。


2. **强制干预 (Intervention)**：
* **替换哪个？** 将模型 A 中你在第一步找到的“英文关键 Head”的输出值，**强行覆盖**到模型 B 对应的位置上。
* **注意**：由于中英文序列长度不同，通常需要对齐关键 Token（例如，把英文 `read` 处的激活值补给中文 `读` 的位置）。


---

#### 第三步：预期结果与数据解读

你需要观察三个指标：

| 状态 | 表现 (F1 Score/Accuracy) | 含义 |
| --- | --- | --- |
| **Baseline** | 极低 | `Corrupted Chinese` 乱序，模型解析完全失败。 |
| **Upper Bound** | 高 | `Clean Chinese` 正常句子的表现。 |
| **Patching Result** | **显著回升** | 如果 F1 恢复到了 Upper Bound 的 70% 以上，说明实验大获成功！ |

**结论的科学描述：**
“当我们将英文句法回路的激活值注入到乱序的中文输入中时，模型恢复了对中文依存关系的识别能力。这证明了模型内部的句法回路具有**跨语言通用性（Language-Agnosticism）**。”

(Path Patching)
---

### 第四步：TodoList（操作指南）

1. **对齐 Token**：确保你的数据集里，中英文的关键成分（主语、谓语、宾语）的索引位置你是知道的。
2. **编写 Hook 函数**：
```python
# 伪代码：强制更改激活值
def patch_head_hook(target_activations, hook, source_activations):
    # 将 source (EN) 的特定激活值补给 target (ZH)
    target_activations[:, head_index, :] = source_activations[:, head_index, :]
    return target_activations

```


3. **运行热力图扫描**：
* 遍历所有的 Layer 和 Head。
* 记录每一次替换后，`Corrupted Chinese` 的解析准确率提升了多少。
* 绘制出一张 **Heatmap**。



---
## 之后再说的内容：
---


## 阶段三：语义解释与验证 (Analysis & Verification)

*目标：解释这些回路到底学到了什么*

1. **QK/OV 矩阵分析**：
* **QK 回路**：这个 Head 在“看”什么？（例如：是否总是英语的“主语位置”看向中文的“主语位置”？）
* **OV 回路**：这个 Head 在“传”什么？（它搬运的是词性信息，还是依存关系标记？）


2. **归纳头 (Induction Heads) 检查**：
* 检查发现的回路中是否存在 Induction Heads，验证迁移是否仅仅是因为模型识别出了重复的结构模式。



---

## 阶段四：干预实验 (Intervention)

*目标：证明回路的因果性*

1. **回路强化训练 (Circuit-Specific Fine-tuning)**：
* 冻结模型 95% 的参数，只允许 Phase 二中发现的 Circuit 权重更新。
* 在辅助任务（如词性标注或简化的句法探测）上训练，观察跨语言迁移 F1 是否进一步提升。


2. **适配器蒸馏 (Adapter Distillation)**：
* 将该 Circuit 的知识提取到一个小的 Adapter 层中，验证是否只需这个小模块就能赋予原模型跨语言能力。



---

## 待办清单 (To-Do List)

| 优先级 | 任务内容 | 预计耗时 |
| --- | --- | --- |
| **P0** | **数据对齐**：准备 100 组中英/英德的对称句法结构数据。 | 2 天 |
| **P0** | **工具适配**：跑通 `HookedTransformer` 加载 XLM-R 权重的 Demo。 | 1 天 |
| **P1** | **Activation Patching**：扫描所有 Layers/Heads，生成恢复率热力图。 | 3-4 天 |
| **P1** | **ACDC 运行**：运行自动化脚本，提取迁移子图。 | 2 天 |
| **P2** | **特征对比**：对比不同语系（如中英 vs. 英德）的迁移回路重合度。 | 3 天 |
| **P3** | **结果汇总与可视化**：绘制回路连接图（Wiring Diagram）。 | 2 天 |

---

### 下一步建议

你可以先从 **Phase 二的第 1 步 (Activation Patching)** 开始。这不需要复杂的 ACDC 算法，只需要用 `TransformerLens` 的 `run_with_hooks` 函数。

**你想让我为你写一个基础的 `Activation Patching` 示例代码框架，好让你直接在 XLM-R 上测试吗？**

Recovery Rate (恢复率公式)：我将改进率改成了 Recovery Rate。这是 Mechanistic Interpretability 领域的标准写法：$$\text{Rate} = \frac{\text{Baseline Loss} - \text{Patched Loss}}{\text{Baseline Loss} - \text{Upper Bound Loss}}$$