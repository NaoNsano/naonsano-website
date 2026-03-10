+++
title = "Testing Post"
date = 2026-03-10T18:34:00+08:00
draft = false
math = true
tags = ["cuda-learning"]
+++

这是 NVIDIA 官方的 CUDA 教程的练习[cuda-training-series](https://github.com/olcf/cuda-training-series)

## 使用共享内存的矩阵乘法

函数调用如下
```cuda
  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
```

核心思路是进行分块（不是线性代数的分块矩阵），因为
$$
C_{ij} = A_i \cdot B_j
$$
而 CUDA 的一大特点是有很多线程。

首先进行分块，每个 block 负责计算 $C$ 的一个相同大小的块（`block_size * block_size`），
接着进入核心思路，由于要使用共享内存，所以每个共享块不能太大，选择同样拷贝相同大小的`A`、`B`分块到`A_s`、`B_s`，一个分块的数值计算可以被视为`A_s`在`A`的行上“滑动”，`B_s`在列上，因为矩阵乘法可以被视为点积，而点积是相加的。

于是可以得出核心思路是使用两个循环，外层循环是前面说的“滑动”，内层循环就是局部的点积。

```cuda
// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();

    }

    // Write to global memory
    C[idy*ds+idx] = temp;
  }
}
```
