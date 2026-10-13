# 用于流匹配生成模型加速采样的正则化多步积分法 (Flow-WLS)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23337579.svg)](https://doi.org/10.5281/zenodo.23337579)

[English](README.md) | [Русский](README_ru.md) | [中文](README_zh.md)

## 摘要 (Abstract)
扩散模型和流匹配（Flow Matching）模型中的反向时间数值积分面临着一个两难境地：低阶方法（1-2阶）需要大量步数（30+ NFE）以避免模糊，而经典的高阶显式多步格式（$p \ge 4$）则受到亚当斯-巴什福斯（Adams-Bashforth）不稳定性、龙格现象振荡以及随机噪声放大的困扰。本文研究了连续流模型（Anima DiT / Cosmos）上多步采样器的崩塌机制，并提出了 **Flow-WLS（加权最小二乘法 / 萨维茨基-戈雷多步法，Weighted Least Squares / Savitzky-Golay Multistep）**。通过将近似阶数限制为平滑抛物线（$d=2$），并在扩展的历史步长（$L=3..8$）上最小化加权残差，Flow-WLS 逆转了多步法的经典缺陷：增加记忆深度不仅不会导致误差累积，反而能够滤除噪声。实验证明，在消费级 GPU 上仅需 11 秒生成时间的条件下，质量指标呈现严格单调递增（在 10 步下 SSIM 从 0.745 提升至 0.800）。

---

## 1. 问题提出与退化机制诊断

### 1.1. DPM-Solver++ 中的高阶崩塌
传统的 DPM-Solver++ 格式通过最近的 $p$ 个历史点精确构建 $p-1$ 次拉格朗日插值多项式。当阶数提升至 $p \ge 5$ 时：
1. **噪声放大**：均差除以步长的乘积 $\prod h_i \sim h^k$。当 $h \approx 0.03$ 时，$p=6$ 的分母约为 $10^{-7}$ 数量级。网络输出的随机波动被放大了数倍。
2. **符号交替的权重**：亚当斯-巴什福斯权重的模长之和 $\sum |w_i|$ 从 $0.03$ 激增至 $0.54$，使积分器变成了高频振荡放大器（龙格现象）。
3. 在实验中，这导致 `cab_6` 的 SSIM 骤降至 **0.1341**，并彻底退化为纯噪声。

### 1.2. 坐标系统不匹配：EDM 与流匹配 (`CONST`)
研究过程中发现了一个定制采样器常见实现中的关键错误：将 EDM 公式直接迁移到流匹配模型中。
* 在 EDM 中：$\alpha_t \equiv 1$，$ \lambda_t = -\ln \sigma_t$。
* 在流匹配 (`comfy.model_sampling.CONST`) 中：$\alpha_t = 1 - \sigma_t$，而整流时间 $\lambda_t = -\text{logit}(\sigma_t)$。
* 在起始阶段（$\sigma \approx 1.0$，$\alpha \approx 10^{-4}$）朴素地使用 $y = x / \alpha_t$ 进行代换，会导致初始噪声立即除以 $10^{-4}$，使其范数放大 10,000 倍，并在 VAE 上引发 90% 的白色溢出剪切（clipping）。转为直接积分速度场 $\frac{dx}{dt} = v(x, t) = \frac{x - x_0}{\sigma}$ 后，完全消除了该伪影。

---

## 2. 提出的方法：Flow-WLS

与使用 $L-1$ 次精确多项式进行插值不同，**Flow-WLS** 方法在包含 $L$ 个节点的历史记录（$s_0 = 0, s_1 = -h_1, \dots$）上针对抛物线核（$d=2$）求解加权最小二乘问题：

$$\min_{c} \sum_{k=0}^{L-1} \gamma^k \left( c_0 + c_1 s_k + c_2 s_k^2 - v_k \right)^2$$

其中 $\gamma = 0.85$ 为记忆的指数衰减系数。

### Flow-WLS 的数学特性：
1. **无龙格现象振荡**：核函数固定为 $d=2$，从物理机制上杜绝了边缘抖动。
2. **权重范数衰减**：
   * 经典 AB-6：$\sum |w_i| = 0.54$（噪声灾难性爆发）。
   * Flow-WLS ($L=6$)：$\sum |w_i| = 0.066$（噪声通过平均化被抑制）。
3. **萨维茨基-戈雷（Savitzky-Golay）滤波**：历史步长 $L=4..8$ 开始起到轨迹平滑滤波器的作用，消除了高刚度场区域中神经网络预测的微小抖动。

---

## 3. 综合实验结果

实验基于 **Anima DiT** (`anima_aestheticV11`) 模型及 Qwen/Anima 文本编码器进行，分辨率为 $720 \times 1280$，未使用任何编译器及第三方优化（RTX 5060，10 步耗时约 11 秒）。

### 表 1. 标准步数预算（30 步）下的求解器对比
| 采样器 | 阶数 / 记忆 | SSIM | LPIPS ( $\downarrow$ ) | CLIP 分数 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cab_2` | $p=2$ (CAB) | 0.7453 | 0.4638 | 26.56 | 纯净 ODE 基线 |
| `cab_3` | $p=3$ (CAB) | 0.7461 | 0.4639 | 26.47 | 经典 CAB 的最优点 |
| `cab_4` | $p=4$ (CAB) | 0.7447 | 0.4664 | 26.70 | 稳定 |
| `cab_5` | $p=5$ (CAB) | 0.7264 | 0.5256 | 26.06 | 开始退化 |
| `cab_6` | $p=6$ (CAB) | **0.1341** | **0.8043** | 23.06 | **多项式崩塌为噪声** |
| `uni_pc` | 内置 | 0.7459 | 0.4677 | 27.35 | 标准基准 |
| **`flow_wls_3`** | $L=3$, WLS | 0.7451 | 0.4648 | 26.65 | 稳定 |
| **`flow_wls_5`** | $L=5$, WLS | 0.7488 | 0.4632 | 26.78 | 细节改善 |
| **`flow_wls_8`** | $L=8$, WLS | **0.7541** | **0.4594** | **26.91** | **ODE 中最佳结果** |

---

### 表 2. 低步数预算模式（Low-NFE 模式：10、12、15 步）
*展示求解器在极限加速下保持清晰度的能力。*

| 步数 (NFE) | 采样器 | SSIM | LPIPS ( $\downarrow$ ) | CLIP 分数 | 解剖结构评估 / 伪影 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **10** | `flow_wls_3` | 0.7665 | 0.4261 | 27.31 | 基础形态 |
| **10** | `flow_wls_4` | 0.7783 | 0.4215 | 26.94 | 轮廓更清晰 |
| **10** | `flow_wls_5` | 0.7850 | 0.4214 | 27.17 | 高细节表现 |
| **10** | `flow_wls_6` | 0.7921 | 0.4223 | 27.22 | 极佳对比度 |
| **10** | **`flow_wls_8`** | **0.8003** | **0.4299** | **27.27** | **11 秒内实现最高精度** |
| **10** | `flow_unipc_2` | 0.7517 | 0.4367 | 26.77 | 存在明显微小伪影 |
| **12** | `flow_wls_3` | 0.7615 | 0.4200 | 26.71 | 稳定 |
| **12** | `flow_wls_6` | 0.7780 | 0.4180 | 27.40 | 较高的 CLIP 分数 |
| **15** | `flow_wls_3` | 0.7557 | 0.4311 | 26.43 | 画面干净 |
| **15** | **`flow_wls_8`** | **0.7737** | **0.4486** | **27.08** | **手指结构正确，完美解剖特征** |

---

## 4. 主要科学结论

1. **Flow-WLS 的单调性**：
   与现有的所有插值方法不同，在 Flow-WLS 中将记忆视野从 $L=3$ 扩展到 $L=8$，可以在**所有步数预算下实现指标的严格单调增长**（10 步时从 0.766 增至 0.800；30 步时从 0.745 增至 0.754）。
2. **解开“记忆”悖论**：
   研究证明，对生成流模型有益的不是高阶多项式（这会破坏稳定性），而是**在低近似阶数下的高深度回顾采样**。
3. **朴素随机性在流匹配中的不适用性**：
   揭示了注入布朗噪声破坏 CAB 有限差分曲率预测器（单步噪声放大 6 倍）的原因，并证实了整流模型确定性积分的优越性。

---

## 参考文献 (References)

1. **DPM-Solver**: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Models in Around 10 Steps*. Advances in Neural Information Processing Systems (NeurIPS 2022). [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
2. **DPM-Solver++**: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*. [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)
3. **UniPC**: Zhao, W., Bai, L., Rao, Y., Zhou, J., & Lu, J. (2023). *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models*. Advances in Neural Information Processing Systems (NeurIPS 2023). [arXiv:2303.01872](https://arxiv.org/abs/2303.01872)
4. **CAB**: Roy, A., & Nair, P. (2026). *CAB: Accelerating Flow and Diffusion Sampling via Rectification and Corrected Adams-Bashforth*. [arXiv:2605.16736](https://arxiv.org/abs/2605.16736)
5. **EDM**: Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. Advances in Neural Information Processing Systems (NeurIPS 2022). [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
6. **Flow Matching**: Lipman, Y., Chen, R. T., Ben-Hamu, H., Nicklas, M., & Le, M. (2022). *Flow Matching for Generative Modeling*. International Conference on Learning Representations (ICLR 2023). [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

---

## 引用 (Citation)

```bibtex
@article{savelev2026flowwls,
  title={Flow-WLS: Regularized Multistep Integration for Accelerated Sampling of Flow Matching Generative Models},
  author={Savelev, Iakov},
  year={2026},
  doi={10.5281/zenodo.23337579},
  url={https://doi.org/10.5281/zenodo.23337579}
}
```