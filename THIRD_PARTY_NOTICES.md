# Third-party format reference

The HPF reader implementation was developed by consulting the BSD-licensed
Biomechanical ToolKit HPF reader:

- Project: Biomechanical ToolKit (BTKCore)
- Source: https://github.com/Biomechanical-ToolKit/BTKCore
- Referenced file: `Code/IO/btkHPFFileIO.cpp`
- BTKCore revision inspected: `d4c03aa9e354be16265d0efe0815c09b35abc642`

The Python reader adds support needed by this dataset, including per-channel
physical sample rates for aligned EMG and IMU signals.
