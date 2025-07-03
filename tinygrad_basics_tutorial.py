#!/usr/bin/env python3

"""
Tinygrad Tutorial: Understanding the Basics
==========================================

This tutorial introduces core Tinygrad concepts for beginners:
- Devices (where computations happen)
- Tensors (the basic data structure)
- Lazy evaluation (delaying computation)
- UOps (computation graphs)

Installation:
1. Clone the repository: git clone https://github.com/tinygrad/tinygrad.git
2. Navigate to directory: cd tinygrad
3. Install in editable mode: python -m pip install -e .
   (The -e flag lets you modify tinygrad source and see changes immediately)
"""

# ============================================
# PART 1: Understanding Devices
# ============================================
# A device is where tensors are stored and computations happen
# Tinygrad automatically detects the best available device (GPU > CPU)

from tinygrad import Device

print("Default device:", Device.DEFAULT)
print()  # This will print "GPU" if you have a GPU, otherwise "CPU"

# ============================================
# PART 2: Creating Your First Tensor
# ============================================
# Tensors are the fundamental data structure in deep learning
# Think of them as arrays that can do math on GPUs

from tinygrad import Tensor, dtypes

# Create a simple tensor with 4 integers
t = Tensor([1, 2, 3, 4])

# Verify the tensor's properties
print("Tensor properties:")
print(f"  Device: {t.device}")     # Where it lives (GPU/CPU)
print(f"  Data type: {t.dtype}")   # int because we used whole numbers
print(f"  Shape: {t.shape}")       # (4,) means 1D array with 4 elements
print()

# ============================================
# PART 3: Lazy Evaluation - The Key Concept
# ============================================
# Unlike NumPy, Tinygrad doesn't compute immediately
# It builds a "recipe" (computation graph) first

print("Printing tensor (notice it doesn't show values):")
print(t)
print()

# The .uop property shows the computation "recipe"
print("The computation graph (UOp) before realization:")
print(t.uop)
print()
# This shows a COPY operation from PYTHON device to GPU/CPU

# ============================================
# PART 4: Realizing Tensors (Executing Computation)
# ============================================
# realize() forces the computation to actually happen

print("Realizing the tensor (executing the computation)...")
t.realize()

print("UOp after realization (now just a BUFFER):")
print(t.uop)
print()

# Tip: Run with DEBUG=2 to see detailed operations
# Example: DEBUG=2 python test.py

# ============================================
# PART 5: Performing Computations
# ============================================
# Let's multiply our tensor by 2

t_times_2 = t * 2  # Creates a new lazy tensor

print("Computation graph for t*2 (before realization):")
print(t_times_2.uop)
print()
# Notice the MUL operation, CONST 2, and broadcasting operations

# Check the result (this forces computation)
result = t_times_2.tolist()
print(f"Result of t*2: {result}")
assert result == [2, 4, 6, 8]
print()

# ============================================
# PART 6: UOp Sharing - An Optimization
# ============================================
# Tinygrad is smart: identical computations share the same graph

print("Creating two identical computations:")
t_times_4_v1 = t * 4
t_times_4_v2 = t * 4

# These tensors are different objects...
print(f"Are tensors the same object? {t_times_4_v1 is t_times_4_v2}")

# ...but they share the same computation graph!
print(f"Do they share the same UOp? {t_times_4_v1.uop is t_times_4_v2.uop}")
print()

# When we realize one...
print("Realizing the first tensor...")
t_times_4_v1.realize()

# ...both get computed (they share the UOp)
print("Second tensor's UOp (also realized):")
print(t_times_4_v2.uop)
print()

# Getting the result doesn't require new computation
print("Getting result from second tensor (no new computation):")
print(f"Result: {t_times_4_v2.tolist()}")

# ============================================
# KEY TAKEAWAYS
# ============================================
"""
1. Tensors: Arrays optimized for GPU computation
2. Devices: Where computations happen (GPU/CPU)
3. Lazy Evaluation: Build computation graph first, execute later
4. UOps: The computation "recipes" that form a graph
5. realize(): Forces actual computation to happen
6. Optimization: Identical computations share the same graph

This lazy approach allows Tinygrad to optimize entire computation
graphs before executing them, making it very efficient!
"""

