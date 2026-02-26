# HPC-Cancer-Modeling

**High-Performance Computing (HPC) for Large-Scale 3D Multiscale Cancer Simulations**

## Overview
This repository implements a spatial multiscale modeling framework for cancer research. The model integrates **discrete agent-based cells** in a 3D environment with **continuum distributions** (such as calcium or nutrient gradients). The implementation focuses on transitioning complex Subcellular Element Models (SEM) into high-performance GPU and CPU code to achieve large-scale biological realism.

For a detailed mathematical background, please refer to the primary publication: "Skin" Paper (Refer to Page 5 for the 3D multiscale model and Pages 19-21 for the Subcellular element model to GPU implementation logic).

---

## Demo: Cell Movement Logic
The current demo version focuses on cell motility driven by interaction forces. The execution pipeline:

1.  **Memory Management:** Initialization and allocation of data structures at both the Host (CPU) and Device (GPU) levels.
2.  **Initial Conditions:** The model reads from `Input/IC` to establish the starting state (typically a single layer of ~60 Type 1 cells).
3.  **Simulation Loop:** * Iterative updates of cell coordinates based on calculated interaction forces.
    * Periodic data output every 10,000 steps for trajectory analysis.
4.  **Finalization:** Memory cleanup and resource deallocation upon reaching the maximum iteration step.

---

## HPC Implementation

### Prerequisites
The environment must be configured with the appropriate compiler and OpenCL drivers. On the Swan cluster, load the following modules:

```bash
module load compiler/gcc/9 cuda/12.2
```

### Compilation
The project includes a Makefile to automate the build process. To compile the code, run:

```bash
make
```

### Job Submission
To execute the simulation on HCC (CPU-based version), use the provided Slurm submission script:

```bash
sbatch cpu.submit
```

---

## Directory Structure

```text
HPC-Cancer-Modeling/
├── src/                # Source code directory
│   ├── main.cpp        # Main entry point
│   ├── kernels.cl      # OpenCL kernels
│   ├── Input/          # Data input directory
│   │   └── IC          # Initial condition file
│   ├── makefile        # Build configuration for GCC and OpenCL
│   └── cpu.submit      # HPC submission scripts
└── Skin.pdf            # Reference paper for the modeling framework
```

---

## Computational Bottlenecks & Optimization Strategies

The current implementation has two primary bottlenecks that limit the scale and speed of the cancer simulations. Addressing these is essential for reaching the target of 10 million level cell simulations.

### 1. Movement Performance: $O(n^2)$ to $O(n)$
The current pairwise interaction calculation leads to a computational complexity of $O(n^2)$, which becomes prohibitive as the cell count increases.

**Proposed Solution:** Implement a **Verlet Grid (Cell Linked List)** method. By partitioning the 3D spatial domain into a grid, interaction forces are only calculated for cells within the same or neighboring grid cells, reducing complexity to $O(n)$.

### 2. Division Performance: CPU-GPU Synchronization
Cell division is currently a serial process executed on the CPU due to the complexity of memory reallocation for new cellular elements.

**Proposed Solution:** Maintain a "Pre-allocated Memory Pool" on the GPU and implement synchronization barriers. By adding synchronization before and after the division function, we can safely update the population count and coordinates without full memory re-initialization. This is particularly critical for modeling aggressive tumor growth where high division rates otherwise create a significant CPU-GPU data transfer bottleneck.
