# PAPO 数学推理强化学习方法复现与训练实践

> 时间：2026.02 - 2026.04  \
> 论文：Stabilizing Rubric Integration Training via Decoupled Advantage Normalization（PAPO）  \
> 链接：https://arxiv.org/abs/2603.26535

围绕大语言模型数学推理强化学习方法 PAPO 开展论文复现与工程实践。该方法旨在解决 GRPO 中**奖励信号耗尽**（ORM 对所有正确答案给出相同信号）与**过程奖励直接引入导致 reward hacking**（PRM-only 易被利用）等问题。

PAPO 的核心做法是将优势函数分解为两个独立归一化的分量：

- $A_{\mathrm{out}}$：基于 ORM（outcome reward model），在**全响应集合**上做 group normalization
- $A_{\mathrm{proc}}$：基于 PRM（process reward model），仅在**正确答案子集**上做归一化（correct-only mask）

最终优势：$A_{\mathrm{total}} = A_{\mathrm{out}} + A_{\mathrm{proc}}$。

---

## 我做了什么

1. 系统阅读并理解 PAPO 论文，梳理 ORM、PRM、GRPO/DAPO、group normalization 与 advantage 计算之间的关系。
2. 参与训练工程实现与调试，重点理解奖励计算、正确答案子集筛选、$A_{\mathrm{out}}/A_{\mathrm{proc}}$ 构造、日志监控与实验脚本组织。
3. 结合代码与实验现象深入理解 decoupled advantage normalization 机制，并对 reward hacking 触发条件与防护手段做对照分析。

## 收获与沉淀

- 熟悉大模型 RL 训练链路与 GRPO/DAPO 优化流程
- 理解 rubric-based PRM 评估方式与 reward hacking 分析方法
- 加深对 RLVR 奖励设计、过程监督、数学推理训练稳定性与工程调试的理解

---

## 复现指南（本仓库）

本仓库基于官方实现（verl / ROLL 生态）组织训练脚本与方法实现，适合按脚本快速跑通 PAPO 与对照实验。

### 环境与硬件

- Python 3.12
- CUDA 12.x
- 多卡训练（原项目配置：8x H100/H200；你也可以按脚本改成更小模型/更少 GPU 进行 smoke test）

说明：训练与 vLLM 服务通常在 Linux 环境更顺畅；如使用 Windows，建议 WSL2 + NVIDIA 驱动/工具链对齐。

### 安装

```bash
# Install (verl backend)
conda create -n papo python=3.12 -y && conda activate papo
pip install vllm==0.14.0 && pip install flash-attn --no-build-isolation
cd verl && pip install -e . && cd .. && pip install -r requirements.txt
```

### 启动 LLM Grader（单独占用 GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 2 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85
```

### 训练

```bash
# Train PAPO
cd verl && bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_dual_lambda1.sh

# Train ORM baseline
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_baseline.sh
```

更多模型（Qwen2.5-3B/14B、Qwen3-4B-Base）与消融（PRM-only、full normalization、multiplicative 等）脚本在 `verl/scripts/`。

---

## 方法备忘（便于对照代码）

```python
# Per group of G responses to the same prompt:
A_out   = normalize(ORM_scores)                    # standard GRPO over all responses
A_proc  = normalize(PRM_scores, mask=correct_only) # among correct responses only
A_total = A_out + A_proc
```

- correct-only normalization：防止错误答案通过高 PRM 分数“钻空子”
- decoupled signals：当组内全部正确时，$A_{\mathrm{out}} \approx 0$，$A_{\mathrm{proc}}$ 仍可提供持续学习信号


---

## Citation

```bibtex
@misc{tan2026stabilizingrubricintegrationtraining,
      title={Stabilizing Rubric Integration Training via Decoupled Advantage Normalization},
      author={Zelin Tan and Zhouliang Yu and Bohan Lin and Zijie Geng and Hejia Geng and Yudong Zhang and Mulei Zhang and Yang Chen and Shuyue Hu and Zhenfei Yin and Chen Zhang and Lei Bai},
      year={2026},
      eprint={2603.26535},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.26535},
}
```

## License

Apache 2.0. Built on [verl](https://github.com/volcengine/verl) and [ROLL](https://github.com/alibaba/ROLL).

## Acknowledgements

[verl](https://github.com/volcengine/verl) | [ROLL](https://github.com/alibaba/ROLL) | [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | [Qwen](https://github.com/QwenLM/Qwen2.5)
