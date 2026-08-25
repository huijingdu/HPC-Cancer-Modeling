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
   * Periodic data output every 100 steps for trajectory analysis.
4.  **Finalization:** Memory cleanup and resource deallocation upon reaching the maximum iteration step.

---

## HPC Implementation

### Prerequisites
The environment must be configured with the appropriate compiler, OpenCL and MPI drivers. On the HPC cluster, load the following modules:

```bash
module load compiler/gcc/9 cuda/12.2 openmpi/4.1
```

### Model Training
The model can be used without training due to the existing `src/rnn/rnn_weights.txt`. To retrain the model, prerequisites must be loaded via:

```bash
module load python/3.12 pytorch/2.5.1
```

Then, run the training:

```bash
cd src/rnn
python3 train/train_rnn.py
```

This trains the model at 6,000 steps with the base features.

### Compilation
The project includes a Makefile to automate the build process. To compile the code, run:

```bash
cd src/
make
```

### Job Submission
To submit a Slurm job, use the included `gpu.submit` script:

```bash
cd src/
sbatch gpu.submit
```

The program screens the environment and uses all of the available GPUs, so defining the number of ranks for a job is done by defining the number of allocated GPUs in the script. The number of tasks per node must equal the number of allocated GPUs on the node. An example with 4 GPUs allocated on 2 nodes:

```Shell
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu --gres=gpu:2
```

---

## Directory Structure

```text
HPC-Cancer-Modeling/
├── src/                    # Source code directory
│   ├── main.c              # Main entry point
│   ├── binning.cl          # OpenCL kernels
│   ├── cellcenter.cl
│   ├── chem_rk_gpu.cl
│   ├── growth.cl
│   ├── movement.cl
│   ├── ovol_fun.cl
│   ├── rnn/                # RNN load balancing controller
│   │   ├── rnn_balancer.c
│   │   ├── rnn_balancer.h
│   │   ├── rnn_weights.txt # Trained weights read by the solver
│   │   └── train/          # Python training scripts
│   │       ├── sim_env.py
│   │       └── train_rnn.py
│   ├── Input/              # Data input directory
│   │   └── IC              # Initial condition file
│   ├── makefile            # Build configuration for GCC and OpenCL
│   └── gpu.submit          # Slurm submission script
└── Skin.pdf                # Reference paper for the modeling framework
```
