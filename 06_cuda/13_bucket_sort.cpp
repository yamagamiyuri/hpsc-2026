#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *bucket, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range){
    bucket[i] = 0;
  }
}

__global__ void count(int *key, int *bucket, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void calc_offset(int *bucket, int *offset, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < range) {
        offset[i] = bucket[i];
    }
    __syncthreads();

    for (int j = 1; j < range; j <<= 1) {
        int temp = 0;
        if (i >= j) temp = offset[i - j];
        __syncthreads();
        if (i >= j) offset[i] += temp;
        __syncthreads();
    }
}

__global__ void write_back(int *key, int *bucket, int *offset, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i < range) {
        int start_pos = offset[i] - bucket[i];
        int num_elements = bucket[i];

        for (int k = 0; k < num_elements; k++) {
            key[start_pos + k] = i;
        }
    }
}


int main() {
  int n = 50;
  int range = 5;
  int threads = 64;
  
  int *key, *bucket, *offset;
  
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&offset, range * sizeof(int));
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  
  printf("\n");
  
  init<<<1, threads>>>(bucket, range);

  int count_blocks = (n + threads - 1) / threads;
  count<<<count_blocks, threads>>>(key, bucket, n);

  calc_offset<<<1, threads>>>(bucket, offset, range);

  write_back<<<1, threads>>>(key, bucket, offset, range);
  
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
}
