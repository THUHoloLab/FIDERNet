<h1> FIDER-Net: imaging through scattering media based on multi-core fiber  </h1>

Turbid fluids are commonly present in biological cavities and can severely impair visualization of luminal surfaces during procedures such as cystoscopy. A Fiber Imaging Descattering and Enhancement Restoration Network (FIDER-Net) is developed by combining statistical-prior-guided preprocessing with multi-stage training. This strategy decouples scattering degradation from turbid liquids and discrete sampling degradation from the multi-core fiber (MCF), enabling restoration under previously unseen scattering concentrations.


## Requirements

FIDER-Net is implemented in Python using PyTorch. GPU inference was verified in the following environment.

- Python 3.9.5, PyTorch >= 1.12.1, NVIDIA Cuda 11.3
- Platforms: Windows 11
- Core image-processing dependencies: NumPy 1.21.6, Pillow 9.5.0, and scikit-image 0.19.3

## Dataset

- The training and testing datasets of FIDER-Net were acquired using a real MCF imaging system. The full datasets will be made publicly available in subsequent updates.




