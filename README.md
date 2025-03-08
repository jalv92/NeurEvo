# NeurEvo

<p align="center">
  🧠
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
- **Unified Interface**: Simple, consistent API for all reinforcement learning tasks
- **Environment Adapters**: Built-in support for Gym, Gymnasium, and custom environments
- **Component Registry**: Extensible architecture for adding custom components
- **Project Integration**: Easy to import and use in any Python project

## Installation

```bash
# From PyPI (recommended)
pip install neurevo

# or clone the repository for development
git clone https://github.com/yourusername/neurevo.git
cd neurevo

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## Quick Start

```python
# Import the framework
import neurevo

# Create a brain with default configuration
brain = neurevo.create_brain()

# Create an agent for a specific environment
agent = brain.create_for_environment("CartPole-v1")

# Train the agent
results = brain.train(episodes=500)

# Evaluate the agent
average_reward = brain.evaluate(episodes=10)
print(f"Average reward: {average_reward}")

# Save the trained agent
brain.save("cartpole_agent.pt")
```

## Using NeurEvo in Your Projects

NeurEvo now features a complete package structure with proper imports, making it easy to incorporate into any Python project.

### Simple Import

```python
# Import the main package
import neurevo

# Create a brain interface - the main entry point
brain = neurevo.create_brain()
```

### Specific Imports

```python
# Import specific components
from neurevo import create_brain
from neurevo.core import NeurEvoAgent, BaseEnvironment
from neurevo.config import NeurEvoConfig
from neurevo.environments import create_custom_environment
```

### Project Integration

1. Install NeurEvo in your Python environment
2. Import the components you need
3. Use the unified API to create and manage agents

```python
# In your project
import neurevo

def train_agent_for_my_project():
    # Create brain interface
    brain = neurevo.create_brain({
        "learning_rate": 0.001,
        "hidden_layers": [256, 128]
    })
    
    # Register your custom environment
    brain.register_environment("MyProjectEnv", my_environment_adapter)
    
    # Create and train agent
    agent = brain.create_for_environment("MyProjectEnv")
    results = brain.train(episodes=1000)
    
    return agent, results
```

## Adapting to Different Environments

NeurEvo can be used with virtually any reinforcement learning environment using our adapter system:

### Custom Environment Integration

```python
from neurevo import create_brain
from neurevo.environments.custom_adapter import create_custom_environment

# Define reset and step functions for your environment
def reset_fn():
    # Reset your environment and return initial state
    return initial_state

def step_fn(action):
    # Execute action and return results
    return next_state, reward, done, info

# Create and register custom environment
brain = create_brain()
brain.register_environment(
    "MyEnvironment",
    create_custom_environment,
    reset_fn=reset_fn,
    step_fn=step_fn,
    observation_shape=(8,),
    action_size=4
)

# Use your custom environment
agent = brain.create_for_environment("MyEnvironment")
brain.train(episodes=500)
```

### Using with Gym/Gymnasium Environments

```python
from neurevo import create_brain

# Create brain interface
brain = create_brain()

# Use any Gym environment
agent = brain.create_for_environment("LunarLander-v2")

# Train the agent
brain.train(agent_id=agent, episodes=1000)
```

### Customizing Neural Architecture

```python
from neurevo import create_brain
from neurevo.config import NeurEvoConfig

# Create custom configuration
config = {
    "hidden_layers": [256, 128, 64],
    "learning_rate": 0.0005,
    "batch_size": 128,
    "curiosity_weight": 0.2
}

# Create brain with custom configuration
brain = create_brain(config)
agent = brain.create_for_environment("MountainCar-v0")
brain.train(episodes=1000)
```

## Project Structure

```
neurevo/
│
├── __init__.py                # Public API exports
├── brain.py                   # Unified interface
│
├── config/                    # Configuration system
├── core/                      # Core agent implementation
├── modules/                   # Neural modules
├── memory/                    # Memory systems
├── learning/                  # Learning algorithms
├── environments/              # Environment adapters
├── utils/                     # Utility functions
│
└── examples/                  # Example environments
```

## Advanced Features

### Skill Transfer Between Environments

```python
# Train in simple environment
brain = create_brain()
agent = brain.create_for_environment("CartPole-v1")
brain.train(episodes=500)

# Save skills
brain.save_skills("cartpole_skills")

# Create new agent for complex environment
new_agent = brain.create_for_environment("BipedalWalker-v3")

# Load skills from previous environment
brain.load_skills("cartpole_skills")
brain.train(episodes=200)  # Faster learning with transferred skills
```

### Customizing Intrinsic Motivation

```python
# Adjust curiosity parameters
config = {
    "curiosity_weight": 0.2,
    "curiosity_type": "rnd",  # Options: "icm", "rnd", "disagreement"
    "novelty_threshold": 0.1
}
brain = create_brain(config)
```

### Dynamic Network Visualization

```python
from neurevo.utils.visualization import visualize_network_growth

# After training
history = brain.visualize_network_growth()
```

## Troubleshooting

### Import Issues

If you encounter import issues:

1. Ensure you've installed the package correctly with `pip install -e .`
2. Try restarting your Python interpreter or IDE
3. Verify your Python path includes the neurevo package

### Testing Imports

You can test if neurevo is correctly installed and importable with:

```python
import neurevo
print(f"NeurEvo version: {neurevo.__version__}")

# Test creating a brain
brain = neurevo.create_brain()
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

## Created BY
- Javier Lora