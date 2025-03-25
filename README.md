# HetRobust

![GitHub stars](https://img.shields.io/github/stars/kenAdamFang/HetRobust?style=social) ![GitHub forks](https://img.shields.io/github/forks/kenAdamFang/HetRobust?style=social) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

Welcome to **HetRobust**, a research-oriented reinforcement learning framework designed to enhance the robustness of heterogeneous multi-agent systems in complex environments. Built upon the robust [PyMARL3 framework](https://github.com/tjuHaoXiaotian/pymarl3), HetRobust integrates seamlessly with SMACv1 and SMACv2 environments, offering researchers and developers an efficient platform for training and evaluation.

---

## ✨ Project Overview

**HetRobust** aims to advance the study of robustness in heterogeneous multi-agent reinforcement learning (MARL). Leveraging the modular and flexible architecture of the [PyMARL3 codebase](https://github.com/tjuHaoXiaotian/pymarl3), this project focuses on training and validating MARL algorithms within the SMACv1 and SMACv2 environments. It provides a foundation for exploring resilience and adaptability in multi-agent settings.

---

## 🚀 Getting Started

### Prerequisites
Before running HetRobust, ensure your environment is properly configured.

1. **Environment Setup**:
   - Execute the following command to install all necessary dependencies:
     ```bash
     source all_pymarl.sh
     ```
   - This script configures the required libraries and tools, ensuring compatibility with the PyMARL3 framework.

2. **Dependency Check**:
   - Verify that Python 3.9 is installed, along with Git and other required packages.

### Running the Training
Once the environment is set up, you can launch HetRobust training and evaluation for SMACv1/v2 using the provided script:

```bash
bash train.sh
