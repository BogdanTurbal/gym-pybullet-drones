# RLALLMA – Reinforcement Learning Augmented Large Language Model Agents

RLALLMA pairs **LLM-driven high-level planning** with **PPO-trained low-level controllers** to coordinate swarms of Crazyflie 2.0 drones in PyBullet.  
It provides a minimal, reproducible research stack for studying how language-conditioned policies and classic reinforcement learning can collaborate in multi-agent robotics.

**Key features**
- Natural-language task specification through prompt-based LLM reasoning  
- Modular PPO controllers fine-tuned per drone role  
- Lightweight Gym-PyBullet-Drones simulation environment wrappers  
- Ready-to-use configs and experiment tracking for rapid benchmarking

**Key files**
- gym_pybullet_drones/examples/llm_orchestrated_multi_drone_demo.py, to run the demo with drones, and set commands through the command line interface after pressing the "set command" button in GUI
- gym_pybullet_drones/examples/rlallma_evaluation.py,  evaluation of the realm


Released under the MIT License
