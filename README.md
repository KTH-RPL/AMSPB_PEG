# Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games

[![arXiv: Learn Controllers for Agile Quadrotors in Pursuit-Evasion Games](https://img.shields.io/badge/arXiv-2407.00000-B31B1B.svg)](https://arxiv.org/abs/2506.02849)

A repository containing the code and resources for the paper **"Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games"** by Alejandro Sánchez Roncero, Yixi Cai, Olov Andersson, and Petter Ögren.

**Abstract**:
The increasing proliferation of small UAVs in civilian and military airspace has raised critical safety and security concerns, especially when unauthorized or malicious drones enter restricted zones. In this work, we present a reinforcement learning (RL) framework for agile 1v1 quadrotor pursuit-evasion. We train neural network policies to command body rates and collective thrust, enabling high-speed pursuit and evasive maneuvers that fully exploit the quadrotor's nonlinear dynamics. To mitigate nonstationarity and catastrophic forgetting during adversarial co-training, we introduce an Asynchronous Multi-Stage Population-Based (AMSPB) algorithm where, at each stage, either the pursuer or evader learns against a sampled opponent drawn from a growing population of past and current policies. This continual learning setup ensures monotonic performance improvement and retention of earlier strategies. Our results show that (i) rate-based policies achieve significantly higher capture rates and peak speeds than velocity-level baselines, and (ii) AMSPB yields stable, monotonic gains against a suite of benchmark opponents.

---

## 🚀 Overview

This project provides a reinforcement learning (RL) framework and neural-network controllers for 1v1 quadrotor pursuit–evasion games. We leverage body-rate control commands and an **Asynchronous Multi-Stage Population-Based (AMSPB)** training loop to achieve agile, high-speed maneuvers and robust adversarial performance.

Key highlights:

* **Body-rate policies** that command roll, pitch, yaw rates and collective thrust to exploit full quadrotor dynamics.
* **AMSPB training** alternates learning between pursuer and evader while sampling from a growing population of past and current policies to mitigate catastrophic forgetting and ensure monotonic improvement.
* **High-fidelity simulation** in NVIDIA Isaac Sim (4.1.0) with realistic quadrotor dynamics at 62.5 Hz.

**Asynchronous Multi-Stage Population-Based (AMSPB)**
<img width="1622" height="415" alt="AMSPB_scheme" src="https://github.com/user-attachments/assets/f28d2783-8f36-4286-9054-561820da791d" />

**RL-Policies**
<img width="1169" height="526" alt="policies" src="https://github.com/user-attachments/assets/e8a47c9e-fa63-4454-8f3a-2a10244486d9" />

---

## 🔧 Installation

### Prerequisites

* **Operating System:** Linux (tested on Ubuntu 24.04)
* **Python:** 3.8 or 3.9
* **Conda:** for environment management
* **OmniDrones** (requires NVIDIA Omniverse and Isaac Sim 4.1.0)

### 1. Install Omniverse and Isaac Sim

1. **Install Omniverse Launcher** following NVIDIA’s instructions:
   [https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install\_workstation.html#isaac-sim-app-install-workstation](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_workstation.html#isaac-sim-app-install-workstation)

2. **Download Isaac Sim 4.1.0** via the Launcher and move it into your Omniverse package folder:

   ```bash
   mv ~/Downloads/IsaacSim-4.1.0 ~/.local/share/ov/pkg
   ```

### 2. Clone this repository

```bash
git clone https://github.com/yourusername/AMSPB_PEG.git
cd AMSPB_PEG
```

### 3. Install OmniDrones

```bash
cd ~/Omnidrones
pip install -e .
```

### 4. Setup Python environment

```bash
conda create -n amspb_env python=3.9 -y
conda activate amspb_env
pip install -r requirements.txt
pip install --upgrade tensordict==0.3.2 torchrl==0.3.1
```

> **Troubleshooting**: If you encounter the error:
>
> ```
> TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'
> ```
>
> Follow the guide: [https://omnidrones.readthedocs.io/en/latest/troubleshooting.html](https://omnidrones.readthedocs.io/en/latest/troubleshooting.html)

---

## 🏁 Quick Start (coming soon)
- Download and test our policies
- Train your own algorithms

## 📈 Results


## 📚 Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{roncero2025learned,
  title={Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games},
  author={Roncero, Alejandro Sanchez and Andersson, Olov and Ogren, Petter},
  journal={arXiv preprint arXiv:2506.02849},
  year={2025}
}
```

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

Alejandro Sánchez Roncero — [alesr@kth.se](mailto:alesr@kth.se)
