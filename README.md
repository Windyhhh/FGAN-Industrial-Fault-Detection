# 🏭 FGAN 工业故障检测 | FGAN Industrial Fault Detection

> **用特征生成对抗网络解决工业故障检测中的数据不平衡问题——少数类故障样本稀缺？FGAN 帮你生成高质量合成样本。**
>
> *Solve data imbalance in industrial fault detection with Feature Generative Adversarial Network — scarce fault samples? FGAN generates high-quality synthetic samples.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🎯 **故障检测** | Fault Detection | 工业设备故障的智能检测与分类 |
| ⚖️ **数据不平衡** | Data Imbalance | 解决正常样本多、故障样本少的核心痛点 |
| 🧠 **特征生成** | Feature Generation | GAN 在特征空间生成合成故障样本 |
| 📈 **性能提升** | Performance Boost | 相比 SMOTE 等传统方法，检测率显著提升 |
| 🏭 **工业场景** | Industrial Scene | 针对工业传感器数据的特性优化 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green?logo=scikit-learn)

---

## 📊 方法对比 | Method Comparison

| 方法 | 少数类召回率 | F1-Score | 训练稳定性 | 计算开销 |
|------|------------|----------|-----------|---------|
| 不处理 (原始数据) | 🔴 低 | 🔴 低 | ✅ 高 | 🟢 低 |
| SMOTE (过采样) | 🟡 中 | 🟡 中 | ✅ 高 | 🟢 低 |
| 欠采样 | 🟡 中 | 🟡 中 | ✅ 高 | 🟢 低 |
| **FGAN (本项目)** | ✅ 高 | ✅ 高 | 🟡 中 | 🟡 中 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/FGAN-Industrial-Fault-Detection.git
cd FGAN-Industrial-Fault-Detection
pip install -r requirements.txt

# 训练 FGAN 生成器
python train_fgan.py --data industrial_data.csv --epochs 500

# 故障检测
python detect.py --model fgan_generator.pt --test test_data.csv
```

---

## 📂 项目结构 | Project Structure

```
FGAN-Industrial-Fault-Detection/
├── train_fgan.py              # FGAN 训练入口
├── detect.py                  # 故障检测入口
├── requirements.txt           # 依赖
├── models/
│   ├── generator.py           # 生成器网络
│   ├── discriminator.py       # 判别器网络
│   └── classifier.py          # 故障分类器
├── data/
│   └── industrial_data.csv    # 工业传感器数据
├── utils/
│   ├── preprocessing.py       # 数据预处理
│   └── evaluation.py          # 评估指标
└── results/                   # 实验结果
```

---

## 🔬 核心原理 | Core Idea

### 特征生成对抗网络 | Feature GAN

```
传统 GAN:  噪声 z → 生成器 G → 合成样本 x' (原始空间)
FGAN:      噪声 z → 生成器 G → 合成特征 f' (特征空间)
                ↓
        特征提取器 (预训练)
                ↓
        真实特征 f
                ↓
        判别器 D 区分真实特征 vs 合成特征
```

### 为什么在特征空间生成？ | Why Feature Space?

1. **维度更低**：特征空间维度远小于原始信号空间，GAN 更容易训练
2. **语义更丰富**：特征空间包含高级语义信息，生成样本更有意义
3. **稳定性更好**：特征空间的分布更平滑，GAN 训练更稳定
4. **分类更准**：生成的特征直接用于分类器训练，效果更好

---

## 🎯 应用场景 | Use Cases

- 🏭 **制造业**：生产线设备的故障检测与预测
- ⚡ **电力系统**：发电机、变压器的故障诊断
- 🚗 **汽车工业**：发动机、变速箱的故障检测
- ✈️ **航空航天**：飞行器部件的健康监测
- 🛢️ **石油化工**：泵、压缩机等旋转机械的故障诊断

---

## 📚 参考文献 | References

- Goodfellow, I., et al. "Generative adversarial nets." NeurIPS 2014.
- Frid-Adar, M., et al. "GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification." Neurocomputing 2018.
- Lei, Y., et al. "Machinery fault diagnosis based on FFT and GAN." Measurement 2021.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **GAN + 工业故障检测的创新方案，Star ⭐ 支持开源！**
