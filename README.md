# PyMARL_LLM

Open-source code for integrating Large Language Models (LLMs) and communication algorithms with Multi-Agent Reinforcement Learning (MARL) using the StarCraft Multi-Agent Challenge (SMAC) as the testing platform. This repository is fine-tuned for SMAC.

## Features

- Integration of LLMs and communication algorithms via replacement of controller and agent files.
- Comprehensive toolkit supporting automatic prompt generation, integration with multiple open-source large models, automatic strategy conversion, and LLM-assisted MARL algorithm training.
- Visualization tools for communication algorithms to assist with algorithm analysis.

## Installation instructions

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl_LLM python=3.8 -y
conda activate pymarl_LLM
bash install_dependencies.sh
```

Set up StarCraft II (2.4.10) and SMAC
```shell
bash install_sc2.sh
```
This will download SC2.4.10 into the `3rdparty` folder and copy the maps necessary to run over.

## Usage

PyMARL_LLM includes implementations of the following algorithms:
- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

### Run an experiment
```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

### Replace Controller and Agent Files
To integrate LLMs or communication algorithms, replace the relevant controller and agent files in the `src/controllers` and `src/agents` directories respectively.

### Communication Algorithm Visualization
To visualize communication data from a specific agent and episode, use the following command:
```shell
python visualization.py --selected_agent 1 --episode 10 --time_step 100
```
Replace `1`, `10`, and `100` with the desired agent ID, episode number, and time step respectively. The visualization script is located in `src/visual`.
