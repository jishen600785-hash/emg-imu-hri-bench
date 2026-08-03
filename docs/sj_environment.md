# Verified environment

- Ubuntu 24.04 under WSL 2
- Python 3.12.3
- NumPy 1.26.4
- SciPy 1.16.3
- scikit-learn 1.7.2
- pandas 2.3.3
- Matplotlib 3.10.7
- PyTorch 2.13.0+cpu

The scikit-learn model is serialized with `joblib`. Load it with a compatible
scikit-learn version. The deep-learning delivery includes TorchScript (`.ts`)
in addition to the Python checkpoint (`.pt`) to simplify inference deployment.
