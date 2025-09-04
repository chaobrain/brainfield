# BrainMass

**Whole-brain modeling with differentiable neural mass models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-brainmass.readthedocs.io-blue.svg)](https://brainmass.readthedocs.io/)

BrainMass is a Python library for whole-brain computational modeling using differentiable neural mass models. Built on JAX for high-performance computing, it provides tools for simulating brain dynamics, analyzing neural networks, and modeling hemodynamic responses.

## Features

- **Neural Mass Models**: Wilson-Cowan model for excitatory-inhibitory population dynamics
- **Hemodynamic Modeling**: BOLD signal simulation using the Balloon-Windkessel model
- **Network Coupling**: Diffusive and additive coupling mechanisms for brain connectivity
- **Noise Modeling**: Ornstein-Uhlenbeck processes for realistic neural noise
- **JAX-Powered**: GPU acceleration and automatic differentiation
- **Real Brain Data**: Example datasets from HCP and other neuroimaging studies

## Installation

### From PyPI (recommended)
```bash
pip install brainmass
```

### From Source
```bash
git clone https://github.com/chaobrain/brainmass.git
cd brainmass
pip install -e .
```

### GPU Support
For CUDA 12 support:
```bash
pip install brainmass[cuda12]
```

For TPU support:
```bash
pip install brainmass[tpu]
```

For whole brain modeling ecosystem:
```bash
pip install BrainX 

# GPU support
pip install BrainX[cuda12]

# TPU support
pip install BrainX[tpu]
```




## Quick Start

### Single Node Simulation

```python
import brainstate
import brainmass
import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
brainstate.environ.set(dt=0.1)

# Create a Wilson-Cowan model with noise
node = brainmass.WilsonCowanModel(
    1,
    noise_E=brainmass.OUProcess(1, sigma=0.01),
    noise_I=brainmass.OUProcess(1, sigma=0.01),
)

# Initialize model states
brainstate.nn.init_all_states(node)

# Define simulation function
def step_run(i):
    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
        return node.update()

# Run simulation
indices = np.arange(10000)
activity = brainstate.transform.for_loop(step_run, indices)

# Plot results
plt.plot(indices * brainstate.environ.get_dt(), activity)
plt.xlabel("Time [ms]")
plt.ylabel("Neural Activity")
plt.show()
```

### Brain Network Simulation

```python
import brainstate
import brainmass
from datasets import Dataset
import numpy as np

# Load brain connectivity data
hcp = Dataset('hcp')

class BrainNetwork(brainstate.nn.Module):
    def __init__(self, signal_speed=2., k=1.):
        super().__init__()
        
        # Get connectivity and distance matrices
        conn_weight = hcp.Cmat
        np.fill_diagonal(conn_weight, 0)
        delay_time = hcp.Dmat / signal_speed
        np.fill_diagonal(delay_time, 0)
        indices_ = np.tile(np.arange(conn_weight.shape[1]), conn_weight.shape[0])
        
        # Create network nodes
        self.node = brainmass.WilsonCowanModel(
            80,  # 80 brain regions
            noise_E=brainmass.OUProcess(80, sigma=0.01),
            noise_I=brainmass.OUProcess(80, sigma=0.01),
        )
        
        # Define coupling between regions
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay('rE', (delay_time.flatten(), indices_), 
                                   init=brainstate.init.Uniform(0, 0.05)),
            self.node.prefetch('rE'),
            conn_weight,
            k=k
        )
    
    def update(self):
        current = self.coupling()
        rE = self.node(current)
        return rE
    
    def step_run(self, i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return self.update()

# Create and run network simulation
net = BrainNetwork()
brainstate.nn.init_all_states(net)
indices = np.arange(0, 6000 // brainstate.environ.get_dt())
brain_activity = brainstate.transform.for_loop(net.step_run, indices)
```


## Examples and Tutorials

The `examples/` directory contains Jupyter notebooks demonstrating:

- **Single Node Dynamics**: Basic Wilson-Cowan model simulation
- **Bifurcation Analysis**: Parameter space exploration
- **Brain Network Modeling**: Whole-brain connectivity simulations
- **Parameter Optimization**: Using Nevergrad for parameter tuning
- **BOLD Signal Analysis**: Hemodynamic response modeling


## Dependencies

Core dependencies:
- `jax`: High-performance computing and automatic differentiation
- `numpy`: Numerical computations
- `brainstate`: State management and neural dynamics
- `brainunit`: Unit system for neuroscience
- `brainscale`: Scaling utilities

Optional dependencies:
- `matplotlib`: Plotting and visualization
- `braintools`: Additional analysis tools
- `nevergrad`: Parameter optimization

## Documentation

Full documentation is available at [brainmass.readthedocs.io](https://brainmass.readthedocs.io/).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use BrainMass in your research, please cite:

```bibtex
@software{brainmass,
  title={BrainMass: Whole-brain modeling with differentiable neural mass models},
  author={BrainMass Developers},
  url={https://github.com/chaobrain/brainmass},
  version={0.0.1},
  year={2024}
}
```

## License

BrainMass is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/chaobrain/brainmass/issues)
- **Documentation**: [ReadTheDocs](https://brainmass.readthedocs.io/)
- **Contact**: chao.brain@qq.com

---

**Keywords**: neural mass model, brain modeling, computational neuroscience, JAX, differentiable programming