# NeurEvo

<p align="center">
  <img src="docs/images/neurevo_logo.png" alt="NeurEvo Logo" width="200"/>
</p>

NeurEvo is a brain-inspired reinforcement learning framework featuring dynamic neural networks that grow and prune connections autonomously. With Hebbian learning, episodic memory, intrinsic curiosity, and skill transfer capabilities, NeurEvo adapts to virtually any reinforcement learning challenge.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Key Features

- **Dynamic Neural Architecture**: Neural networks that grow or shrink based on task complexity
- **Hebbian Learning**: Biologically-inspired learning where "neurons that fire together, wire together"
- **Episodic Memory**: Stores and analyzes complete experiences with causal reasoning
- **Intrinsic Curiosity**: Self-motivated exploration of novel or uncertain states
- **Skill Transfer**: Reuse knowledge between different tasks and environments
- **Meta-Learning**: Automatic hyperparameter optimization during training
- **Modular Design**: Easy to adapt to any reinforcement learning problem

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurevo.git
cd neurevo

# Install with pip
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from neurevo.core.agent import NeurEvoAgent
from neurevo.config.config import NeurEvoConfig
import gym

# Create environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create agent with default configuration
config = NeurEvoConfig()
agent = NeurEvoAgent(state_size, action_size, config)

# Training loop
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get action from agent
        action = agent.act(state)
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Let agent process experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train agent periodically
        if episode > 10:  # Start training after collecting some experiences
            agent.train()
            
        state = next_state
        total_reward += reward
    
    # Update agent's learning strategy
    agent.update_meta_learning(env, total_reward, 1 if total_reward > 195 else 0)
    
    # Decay exploration rate
    agent.decay_epsilon()
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")

# Save the trained agent
agent.save("cartpole_agent.pt")
```

## Adapting to Different Environments

NeurEvo can be used with virtually any reinforcement learning environment. Here's how to integrate it with different types of environments:

### Custom Environment Integration

Any environment that provides the standard RL interface (reset, step) can work with NeurEvo:

```python
class MyCustomEnvironment:
    def __init__(self):
        self.state_size = 8  # Size of state representation
        self.action_size = 4  # Number of possible actions
        # Initialize environment state
        
    def reset(self):
        # Reset environment to initial state
        return initial_state
        
    def step(self, action):
        # Execute action and return results
        next_state = ...
        reward = ...
        done = ...
        return next_state, reward, done

# Create agent for this environment
env = MyCustomEnvironment()
agent = NeurEvoAgent(env.state_size, env.action_size)
```

### Adapting PyTorch Models

You can customize the neural architecture for specific domains:

```python
from neurevo.modules.dynamic_layer import DynamicLayer
import torch.nn as nn

# Custom perception module for image-based states
class ImagePerceptionModule(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dynamic_layer = DynamicLayer(3136, output_size)  # 3136 = flattened conv output
        
    def forward(self, x):
        conv_features = self.conv_layers(x)
        return self.dynamic_layer(conv_features)

# Then integrate with agent:
agent.modules['perception'] = ImagePerceptionModule(4, 64).to(agent.device)
```

### Using with Gym Environments

NeurEvo works seamlessly with OpenAI Gym:

```python
import gym
from neurevo import create_agent, train_agent

# Create environment
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create agent
agent = create_agent(state_size, action_size)

# Train agent
train_agent(agent, env, num_episodes=1000, save_path="lunar_lander_agent.pt")
```

## Project Structure

```
neurevo/
│
├── __init__.py                # Public API exports
├── neurevo_main.py            # Main orchestrator
│
├── config/                    # Configuration system
├── core/                      # Core agent implementation
├── modules/                   # Neural modules
├── memory/                    # Memory systems
├── learning/                  # Learning algorithms
├── utils/                     # Utility functions
│
└── examples/                  # Example environments
```

## Advanced Features

### Skill Transfer Between Environments

```python
# Train in simple environment
agent = create_agent(state_size, action_size)
train_agent(agent, simple_env, num_episodes=500)

# Transfer to more complex environment
complex_env = ComplexEnvironment()
agent.transfer_knowledge(complex_env)
train_agent(agent, complex_env, num_episodes=200)
```

### Customizing Intrinsic Motivation

```python
# Adjust curiosity parameters
config = NeurEvoConfig()
config.CURIOSITY_WEIGHT = 0.2  # Stronger intrinsic motivation
config.NOVELTY_THRESHOLD = 0.1  # More sensitive to novel states
agent = create_agent(state_size, action_size, config)
```

### Dynamic Network Visualization

```python
from neurevo.utils.visualization import visualize_network_growth

# After training
history = visualize_network_growth(agent)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by principles from neuroscience and cognitive science
- Built with PyTorch and compatible with OpenAI Gym environments