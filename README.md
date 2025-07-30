# AMSPB\_PEG

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
pip install --upgrade tensordict==0.3.2 torchrl==0.3.1
pip install -r requirements.txt
```

> **Troubleshooting**: If you encounter the error:
>
> ```
> TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'
> ```
>
> Follow the guide: [https://omnidrones.readthedocs.io/en/latest/troubleshooting.html](https://omnidrones.readthedocs.io/en/latest/troubleshooting.html)

---

## 🏁 Quick Start

1. **Configure experiment**: edit `configs/your_experiment.yaml` to set training hyperparameters, arena size, reward weights, etc.
2. **Train a policy**:

   ```bash
   python train.py --agent pursuer --config configs/pursuer.yaml
   ```
3. **Evaluate and benchmark**:

   ```bash
   python evaluate.py --checkpoint runs/pursuer/latest.pt --benchmarks hover circular repel
   ```
4. **Visualize results** (coming soon): use the `scripts/visualize.py` to plot capture rates and trajectories.

---

## 📈 Results

<!-- ## Placeholder for performance plots -->

*Concluded in the paper that rate-based policies achieve:*

* Up to **12.90 m/s** peak linear speed vs. 10.49 m/s for velocity-based.
* Up to **12.85 rad/s** peak angular rate vs. 6.38 rad/s for velocity-based.
* **Monotonic capture-rate improvement** against a suite of benchmark opponents using AMSPB.

!

*> Section V of the paper presents detailed tables and ablation.*

---

## 📚 Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{roncero2025learnedquadpeg,
  title     = {Learned Controllers for Agile Quadrotors in Pursuit–Evasion Games},
  author    = {S{'a}nchez Roncero, Alejandro and Andersson, Olov and {
 O}gren, Petter},
  journal   = {arXiv preprint arXiv:2407.00000},
  year      = {2025},
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

Alejandro Sánchez Roncero — [alesr@kth.se](mailto:alesr@kth.se)
Project Link: [https://github.com/yourusername/AMSPB\_PEG](https://github.com/yourusername/AMSPB_PEG)
