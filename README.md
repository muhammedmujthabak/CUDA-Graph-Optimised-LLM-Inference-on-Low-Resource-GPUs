# CUDA Graph Optimised LLM Inference on Low-Resource GPUs

An LLM inference optimization project that uses **CUDA Graphs and StaticCache** to reduce CPU and kernel launch overhead during decoding. The project evaluates the performance of CUDA Graph replay against standard eager execution on a low-resource **NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM**.

## Technologies Used

* Python
* PyTorch
* CUDA / CUDA Graphs
* Hugging Face Transformers
* Qwen2.5-0.5B
* StaticCache
* Streamlit
* Matplotlib

## Features

* CUDA Graph capture and replay for LLM decoding
* Static KV-cache for fixed GPU memory addresses
* Eager vs CUDA Graph performance comparison
* Decode latency and throughput benchmarking
* Step-by-step correctness validation
* Interactive Streamlit inference interface
* Designed for low-resource GPU environments

## Project Workflow

```text
Load LLM
   ↓
Prepare Static KV Cache
   ↓
Warm-up Decode
   ↓
Capture CUDA Graph
   ↓
Replay CUDA Graph
   ↓
Generate Tokens
   ↓
Benchmark Performance
   ↓
Validate Correctness
```

## Problem & Solution

### Problem

Standard eager-mode LLM inference can introduce significant CPU and kernel launch overhead during token-by-token decoding. Dynamic KV-cache management can also create memory-address changes that are incompatible with CUDA Graph replay.

### Solution

The project replaces the dynamic KV cache with **StaticCache**, which pre-allocates GPU memory at fixed addresses. The decoding operation is then captured using **CUDA Graphs** and replayed for subsequent tokens, reducing repeated CPU launch overhead.

## Results

The CUDA Graph implementation achieved significant decoding performance improvements compared with eager execution.

| Decode Length |          Eager |    CUDA Graph | Speedup |
| ------------: | -------------: | ------------: | ------: |
|            32 | 73.60 ms/token | 8.49 ms/token |   8.67× |
|            64 | 38.74 ms/token | 8.64 ms/token |   4.48× |
|           128 | 35.87 ms/token | 8.41 ms/token |   4.27× |
|           256 | 40.84 ms/token | 8.74 ms/token |   4.67× |

## What I Learned

* How CUDA Graphs reduce CPU and kernel launch overhead.
* How KV-cache management affects LLM inference performance.
* Why dynamic memory allocation can cause CUDA Graph correctness issues.
* How StaticCache provides stable GPU memory addresses.
* How to benchmark LLM inference using latency, throughput, and speedup.
* How to compare eager and CUDA Graph execution for correctness.

## Future Improvements

* Support additional LLM architectures and model sizes.
* Evaluate performance across different GPU configurations.
* Explore INT8 and 4-bit quantized inference.
* Improve support for different sequence lengths.
* Add automated performance and correctness testing.
* Explore optimized model-serving and deployment techniques.

## Hardware & Software

**Hardware**

* NVIDIA RTX 3050 Laptop GPU
* 4 GB VRAM
* Windows 11

**Software**

* Python
* PyTorch
* CUDA Toolkit
* Hugging Face Transformers
* Streamlit

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/muhammedmujthabak/CUDA-Graph-Optimised-LLM-Inference-on-Low-Resource-GPUs.git
cd CUDA-Graph-Optimised-LLM-Inference-on-Low-Resource-GPUs
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

### 5. Run the benchmark

```bash
python benchmark_extended.py
```

### 6. Run correctness validation

```bash
python validate.py
```

## Project Structure

```text
CUDA-Graph-Optimised-LLM-Inference-on-Low-Resource-GPUs/
│
├── app.py
├── benchmark_extended.py
├── validate.py
├── requirements.txt
├── README.md
│
├── models/
│   └── static_memory.py
│
├── benchmarks/
│   └── results/
│
└── assets/
```

## License

This project is developed for educational and research purposes.
