<div align="center">

# 🚗 Day-to-Day Route–Departure Choice Dynamics

### Two-Stage Bayesian Perception Updating under Alternative Information Structures

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)]()

</div>

---

## 📖 Overview

This repository implements a **day-to-day (DTD) traffic simulation framework** for studying how travelers learn and adapt their route and departure-time choices under imperfect travel information.

### Key Contributions

| Component | Description |
|-----------|-------------|
| **Two-Stage Bayesian Updating** | Travelers integrate pre-trip broadcast info and post-trip experience |
| **SRD Choice Model** | Simultaneous route and departure-time choice with schedule delay penalties |
| **Information Structures** | Compare no-info, historical broadcast, and peer-to-peer sharing |
| **Disruption Analysis** | Study network resilience under capacity reduction scenarios |

---

## 🏗️ Project Structure

```
dtd_paper/
├── dtd_srdt/                    # Core simulation library
│   ├── common.py                # Bayesian belief, BPR function, utilities
│   ├── toy_sim.py               # Toy network simulation
│   ├── sioux_sim.py             # Sioux Falls network simulation
│   ├── metrics.py               # Performance metrics
│   └── plotting.py              # Visualization utilities
├── bdi_test/                    # Experiment scripts
│   ├── baseline1_srdt_bayes_toy.py
│   ├── baseline1_srdt_bayes_sioux.py
│   └── paper_figs.py            # Generate paper figures
├── run_all_experiments.py       # Main experiment runner
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Perseus1993/dtd_paper.git
cd dtd_paper
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run all experiments and generate figures
python run_all_experiments.py

# Run specific experiment
python bdi_test/baseline1_srdt_bayes_sioux.py --scenario S1 --days 100
```

---

## 📊 Information Scenarios

| Scenario | Name | Description |
|:--------:|------|-------------|
| **S0** | No Information | Experience-only learning |
| **S1** | Historical Broadcast | System-wide historical travel times |
| **S2** | Peer Sharing | Share-dependent reliability based on choice proportion |

---

## 📈 Key Findings

> **Efficiency–Stability Trade-off**: Reliable broadcast information mitigates peak congestion but may induce stronger day-to-day oscillations. Share-dependent information can delay equilibrium recovery due to herding effects.

---

## 📝 Citation

```bibtex
@article{li2026dtd,
  title={Day-to-Day Route--Departure Choice Dynamics with Two-Stage 
         Bayesian Perception Updating under Alternative Information Structures},
  author={Li, Gen and Xu, Pengcheng and Lan, Jieyuan},
  journal={Under review},
  year={2026}
}
```

---

## 📬 Contact

**Gen Li** - gen_li1993@163.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
