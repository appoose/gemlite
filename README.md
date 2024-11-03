# GemLite

<img src="images/gemlite%20banner.png" alt="GemLite Logo" width="200">
**[GemLite](https://github.com/mobiusml/gemlite/)** is a collection of straightforward CUDA and Triton kernels for efficient, fused low-bit matrix multiplication. It is specifically designed for **simplicity** and **reusability**. This project began as a way to address the challenges we faced in customizing existing low-bit kernels.

GemLite provides both **flexibility** and **performance**, enabling users to easily modify the codebase to develop high-performance kernels tailored to their specific needs. The project started with CUDA kernels, but we've added Triton kernels to enhance flexibility. Major recent improvements are in the [Triton branch](https://github.com/appoose/gemlite/tree/master/gemlite/triton_kernels). Our goal is to support various combinations of bitrates for weights and activations. Different GPU architectures (e.g., RTX 4090, A100, H100) have their own optimal configurations, so we provide a range of kernels and auto-tuning features to select the best for each setup.

### Recent Highlights

- **New GEMV RevSplitK Algorithm**: Outperforms GEMM Split-K and GEMV for batch-size=1.
- **Channel-wise Scaling**: Added support for channel-wise scaling for weights, activations, and both.
- **Precision Support**: Includes FP8 x FP8, FP8 x Wn, and INT8 x Wn.
- **Improved Autotune Speed**.
- **Optimized Configurations**: Enhancements for 4090 RTX, A100, and H100.
- **torch.compile() Support**.
- **Tunable A Loading Order, Eviction Policies, and Atomic Add Mode**.

### Performance at glance ### 
-suggest we add figures for a matrix multiplication ( in 3  different GPUS ? )
- prefill phase and decoding time ( when the benchmark is stable )

While GemLite can outperform the best existing implementations on large matrices, there's still potential for further optimization!

## Getting Started

### Installation

```sh
pip install git+https://github.com/mobiusml/gemlite/
```

### Basic Usage

```python
from gemlite.core import DType, GemLiteLinear, set_autotune

# Set autotuner (enabled for group_size < 128)
set_autotune({'GEMV_REVSPLITK': True, 'GEMV': True, 'GEMM_SPLITK': True, 'GEMM': True},
             exhaustive=False,  # Enable exhaustive search for best kernel.
             use_cuda_graph=False)  # Use CUDA Graphs for benchmarking.

# Using Triton backend as default
gemlite_linear = GemLiteLinear(
    W_nbits=8,  # weight quantization bitwidth (supported: 8, 4, 2, 1)
    group_size=128,  # enable autotune for group_size < 128
    in_features=4096,
    out_features=4096,
    input_dtype=DType.FP16,
    output_dtype=DType.FP16,
    scaled_activations=False,
)

# Pack the model weights and biases
gemlite_linear.pack(W_q, scales, zeros, bias)

# Forward pass
out = gemlite_linear(x)
```

For more examples, check out the [examples folder](https://github.com/mobiusml/gemlite/tree/master/examples). Install dependencies with `./install_dependencies.sh` before running.

## Deep Dive

### Triton Kernels

We provide multiple versions of Triton kernels, each designed for different scenarios:

- **[GEMV](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_A16fWnO16f_int32packing.py)**: Splits activations into chunks for small batch sizes (M < 16).
- **[GEMM](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_A16fWnO16f_int32packing.py)**: Optimized with tensor cores for larger batches.
- **[GEMM Split-K](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_splitK_A16fWnO16f_int32packing.py)**: Optimized for batched LLM decoding.
- **[GEMV RevSplit-K](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py)**: New algorithm for best performance in batch-size=1 decoding.

All kernels support 8, 4, 2, and 1-bit precisions.

### CUDA Implementation

Our CUDA implementation adapts methods from fast GEMV implementations such as [Bruce-Lee-LY's implementation](https://github.com/Bruce-Lee-LY/cuda_hgemv) and [FastGEMV](https://github.com/wangsiping97/FastGEMV), leveraging shared memory and warp reduction for performance gains. For more details, refer to our [blog post](https://mobiusml.github.io/gemlite_blogpost/).

### Available Kernels

#### Triton Kernels

- **FP8 x FP8** - with grouping
- **FP8 x Wn** - with grouping
- **INT8 x Wn** - with grouping
- **A16W8 (GEMV + GEMM)** - with grouping
- **A16W4 (GEMV + GEMM)** - with grouping
- **A16W2 (GEMV + GEMM)** - with grouping
- **A16W1 (GEMV + GEMM)** - with grouping

#### CUDA Kernels

- **A16W8 (GEMV - batch-size=1)** - no grouping
- **A16W4 (GEMV - batch-size=1)** - no grouping
- **A16W2 (GEMV - batch-size=1)** - no grouping
- **A8W8 (GEMV - batch-size=1)** - no grouping
- **A8W4 (GEMV - batch-size=1)** - no grouping
- **A8W2 (GEMV - batch-size=1)** - no grouping

### Limitations

- Performance could be improved for smaller matrices or low batch sizes.
- Launching Triton kernels can have high overhead for light workloads.
- Autotuning is time-consuming; consider adding more configurations for different devices.
- Optimized mainly for RTX 4090.

## Performance

We've benchmarked our kernels across various bitrates and batch sizes on RTX 4090, showing significant speed-ups relative to FP16 `torch.matmul`. Results are reproducible using `examples/benchmark_triton.py`.

## Citation

If you use GemLite in your research, please cite us:

```bibtex
@misc{badri2024gemlite,
  title  = {Gemlite: Towards Building Custom Low-Bit Fused CUDA Kernels},
  url    = {https://github.com/mobiusml/gemlite},
  author = {Hicham Badri, Appu Shaji},
  month  = {August},
  year   = {2024}
}
```

---

Feel free to explore the code, contribute, or provide feedback. We're constantly looking for ways to improve and expand this project!
