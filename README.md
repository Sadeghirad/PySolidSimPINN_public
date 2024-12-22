# PySolidSimPINN
## Overview
This repository contains the implementation of the PySolidSimPINN, a novel approach to enhance the convergence of physics-informed neural networks (PINNs) for solid mechanics problems using a rigid-body motion reduction (RBMR) technique.

## Features
- **Standard PINN:** Implements the standard physics-informed neural network approach.
- **Rigid-Body Motion Reduction (RBMR):** Incorporates a modified collocation loss function to eliminate rigid-body motion analytically.
- **Validation:** Validated on numerical experiments including 2D elastostatic problems such as pure bending cantilever beam and curved cantilever beam under end loading.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the simulation for pure bending cantilever beam, execute:
```bash
python ./pure_bending/tests/example_pure_bending_beam.py
```
To run the simulation for curved cantilever beam, execute:
```bash
python ./curved_beam/tests/example_curved_beam.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
