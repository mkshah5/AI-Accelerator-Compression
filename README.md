# Code from ***A Portable, Fast, DCT-based Compressor for AI Accelerators***
## Overview
This is the repository for the paper [***A Portable, Fast, DCT-based Compressor for AI Accelerators***](https://dl.acm.org/doi/10.1145/3625549.3658662). The DCT+Chop compressor has been tested on five systems: NVIDIA A100 GPU, Cerebras CS-2, SambaNova SN30, Groq Groqchip, and Graphcore IPU. While this work originally targets training data, the DCT+Chop compressor can be used for any data on these systems. We use benchmarks from [SciML-bench](https://github.com/stfc-sciml/sciml-bench?tab=readme-ov-file#23-benchmarks-and-datasets).

## Versions Tested
- PyTorch 2.0.1
- Cerebras Release 2.0.1
- SambaFlow 1.17
- GroqFlow 4.2.1
- PopTorch (Graphcore) 3.3.0

## Directory Overview
- comp_benchmark: directory containing benchmarking scripts to test scaling input data size for the compressor
- compressor: contains entry points for the compressor
- emdenoise: EMDenoise SciML-Bench benchmark
- opticaldamage: OpticalDamage SciML-Bench benchmark
- resnet34: ResNet34 with CIFAR10 dataset benchmark
- slstrcloud: SLSTRCloud SciML-Bench benchmark
- slstrcloud_highres: SLSTRCloud SciML-Bench benchmark with higher resolution data
- utils: utility functions

## File Overview
All scripts to run code need a `config.txt` file. See `emdenoise/config-ch4.txt` for an example.

In comp_benchmark, each filename to test compression follows the `bench_compress_<platform>.py` format, while decompression tests follow the `bench_<platform>.py` format.

Example (SambaNova):
```
python bench_samba.py --config_path="./config.txt" --num-iterations=10 --compressor="dct"
```

Each network (emdenoise, opticaldamage, resnet34, slstrcloud) has several scripts formatted as `<network>_<platform>.py`. Remember to pass the config file for the compressor.

Example (Graphcore):
```
cd resnet34
python resnet34_graphcore.py --config_path="./config-ch4.txt" --num-iterations=1 --num-epochs=30
```

## Citation
Please use the citation below if you reference this work:

Milan Shah, Xiaodong Yu, Sheng Di, Michela Becchi, and Franck Cappello. 2024. A Portable, Fast, DCT-based Compressor for AI Accelerators. In Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing (HPDC '24). Association for Computing Machinery, New York, NY, USA, 109–121. https://doi.org/10.1145/3625549.3658662
