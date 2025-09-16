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

### Ecosystem

For whole brain modeling ecosystem:
```bash
pip install BrainX 

# GPU support
pip install BrainX[cuda12]

# TPU support
pip install BrainX[tpu]
```


## Dependencies

Core dependencies:
- `jax`: High-performance computing and automatic differentiation
- `numpy`: Numerical computations
- `brainstate`: State management and neural dynamics
- `brainunit`: Unit system for neuroscience
- `brainscale`: Online learning support
- `braintools`: Additional analysis tools

  Optional dependencies:
- `matplotlib`: Plotting and visualization
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
  year={2025}
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