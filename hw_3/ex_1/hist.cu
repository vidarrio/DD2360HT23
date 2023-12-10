#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ unsigned int private_histo[NUM_BINS];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < num_bins / blockDim.x; i++) {
    if (tid + i * blockDim.x < num_bins) {
      private_histo[tid + i * blockDim.x] = 0;
    }
  }
  __syncthreads();

  if (gid < num_elements) {
    atomicAdd(&(private_histo[input[gid]]), 1);
  }

  __syncthreads();

  for (int i = 0; i < num_bins / blockDim.x; i++) {
    if (tid + i * blockDim.x < num_bins) {
      atomicAdd(&(bins[tid + i * blockDim.x]), private_histo[tid + i * blockDim.x]);
    }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < num_bins) {
    if (bins[gid] > 127) {
      bins[gid] = 127;
    }
  }
}

int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("Usage: histogram <inputLength>\n");
    exit(1);
  }

  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, NUM_BINS - 1);
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = distribution(generator);
  }

  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
  for (int i = 0; i < NUM_BINS; i++) {
    resultRef[i] = 0;
  }

  for (int i = 0; i < inputLength; i++) {
    if (resultRef[hostInput[i]] < 127) {
      resultRef[hostInput[i]]++;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  int threadsPerBlock = 1024;
  int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

  dim3 blockDim(threadsPerBlock, 1, 1);
  dim3 gridDim(blocksPerGrid, 1, 1);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridDim, blockDim>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  blocksPerGrid = (NUM_BINS + threadsPerBlock - 1) / threadsPerBlock;

  dim3 blockDim2(threadsPerBlock, 1, 1);
  dim3 gridDim2(blocksPerGrid, 1, 1);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridDim2, blockDim2>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      printf("Mismatch at %d: %d vs %d\n", i, hostBins[i], resultRef[i]);
      exit(1);
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

