
#include <stdio.h>
#include <sys/time.h>
#include <chrono>

#define DataType double

using namespace std::chrono;

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * numBColumns + col;
  
  

  if (row < numARows && col < numBColumns) {
    for (int i = 0; i < numAColumns; i++) {
      C[index] += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
  }
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows * numAColumns; i++) {
    hostA[i] = (DataType)rand()/(DataType)RAND_MAX;
  }
  
  for (int i = 0; i < numBRows * numBColumns; i++) {
    hostB[i] = (DataType)rand()/(DataType)RAND_MAX;
  }

  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      for (int k = 0; k < numAColumns; k++) {
        resultRef[i * numCColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
 
  //@@ Initialize the grid and block dimensions here
  
  int n = numARows*numBColumns;
  int block_size = 32;
  int grid_size = ceil(n/block_size);

  dim3 threadsPerBlock(block_size, block_size, 1);

  dim3 blocksPerGrid(grid_size, grid_size, 1);

  //@@ Launch the GPU Kernel here
  gemm<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numCRows * numCColumns; i++) {
    if (abs(hostC[i] - resultRef[i]) > 1e-5) {
      printf("Error: hostC[%d] = %f, resultRef[%d] = %f\n", i, hostC[i], i, resultRef[i]);
      break;
    }
  }


  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here

  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
