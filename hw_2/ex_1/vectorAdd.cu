
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <cuda_profiler_api.h>

#define DataType double

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here

  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < len) {
    out[id] = in1[id] + in2[id];
  }
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("Usage: vecAdd inputLength\n");
    exit(1);
  }

  inputLength = atoi(argv[1]);
  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));

  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = (DataType) rand() / RAND_MAX;
    hostInput2[i] = (DataType) rand() / RAND_MAX;
  }

  resultRef = (DataType *) malloc(inputLength * sizeof(DataType));

  for (int i = 0; i < inputLength; i++) {
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ start cuda profiler
  cudaProfilerStart();

  //@@ start cpu timer
  double cpu_start = get_wall_time();

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here

  // 1024 is the maximum number of threads per block for compute capability 6.1
  int threadsPerBlock = 1024;

  // round up to the nearest integer so we don't have a partial block
  int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Blocks per grid: " << blocksPerGrid << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;

  dim3 dimGrid(blocksPerGrid, 1, 1);
  dim3 dimBlock(threadsPerBlock, 1, 1);

  //@@ Launch the GPU Kernel here
  vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ stop cpu timer
  double cpu_end = get_wall_time();

  //@@ stop cuda profiler
  cudaProfilerStop();

  //@@ Insert code below to compare the output with the reference
  int errors = 0;
  for (int i = 0; i < inputLength; i++) {
    if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
      errors++;
    }
  }

  std::cout << "Error: " << errors << std::endl;

  //@@ Insert code below to print out the timing results
  std::cout << "Execution time: " << cpu_end - cpu_start << std::endl;

  //@@ Free the GPU memory here

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  
  return 0;
}
