<div align="center">

# 🏭 FGAN-Industrial-Fault-Detection

### FGAN + KDE industrial process fault detection.

Fault detection on the Tennessee Eastman Process (TEP) with FGAN and KDE — 21 fault types, fully visualized.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

**FGAN-Industrial-Fault-Detection** applies **FGAN** (with **KDE**) to industrial process fault detection on the **Tennessee Eastman Process (TEP)** benchmark — covering **21 fault types** with full visualization.

> [!NOTE]
> 中文项目：FGAN + KDE 工业过程故障检测——田纳西伊斯曼过程（TEP），21 种故障，完整可视化。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/FGAN-Industrial-Fault-Detection.git
cd FGAN-Industrial-Fault-Detection

pip install -r requirements.txt

# run the detection pipeline (see README / scripts in repo)
python src/main.py
```

The TEP dataset (`d00_train.csv`, `d00_test.csv` … `d20_test.csv`) ships in `data/`.

---

## Features

- **FGAN + KDE** — generative + kernel-density fault detection.
- **TEP benchmark** — standard 21-fault Tennessee Eastman data.
- **Full visualization** — detection results and distributions.

---

## Project Structure

```
FGAN-Industrial-Fault-Detection/
├── data/                 # TEP train/test CSVs (d00–d20)
├── src/                  # detection pipeline
└── README.md
```

---

## License

MIT — free to use, modify and distribute.
