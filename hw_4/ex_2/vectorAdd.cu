
#include <stdio.h>
#include <sys/time.h>
#include <iostream>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    //@@ Insert code to implement vector addition here

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < len) {
        printf("id: %d = %f\n", id, in1[id] + in2[id]);
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

    int S_seg = 4;

    //@@ Insert code below to read in inputLength from args

    if (argc != 2) {
        printf("Usage: vecAdd inputLength\n");
        exit(1);
    }

    inputLength = atoi(argv[1]);

    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);


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

    //@@ Insert code below to allocate GPU memory here

    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    //@@ Insert code to create cuda streams here
    cudaStream_t stream[S_seg];

    for (int i = 0; i < S_seg; i++) {
        cudaStreamCreate(&stream[i]);
    }

    //@@ Insert code to below to Copy memory to the GPU here

    for (int i = 0; i < S_seg; i++) {
        if (i == S_seg - 1) {
            cudaMemcpyAsync(deviceInput1 + i * inputLength / S_seg, hostInput1 + i * inputLength / S_seg, (inputLength / S_seg + inputLength % S_seg) * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(deviceInput2 + i * inputLength / S_seg, hostInput2 + i * inputLength / S_seg, (inputLength / S_seg + inputLength % S_seg) * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
        } else {
            cudaMemcpyAsync(deviceInput1 + i * inputLength / S_seg, hostInput1 + i * inputLength / S_seg, inputLength / S_seg * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(deviceInput2 + i * inputLength / S_seg, hostInput2 + i * inputLength / S_seg, inputLength / S_seg * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
        }
    }

    //@@ Initialize the 1D grid and block dimensions here

    // 1024 is the maximum number of threads per block for compute capability 6.1
    int threadsPerBlock = 10;

    // round up to the nearest integer so we don't have a partial block
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock << std::endl;

    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    //@@ Launch the GPU Kernel here

    for (int i = 0; i < S_seg; i++) {
        if (i == S_seg - 1) {
            vecAdd<<<dimGrid, dimBlock, 0, stream[i]>>>(deviceInput1 + i * inputLength / S_seg, deviceInput2 + i * inputLength / S_seg, deviceOutput + i * inputLength / S_seg, inputLength / S_seg + inputLength % S_seg);
        } else {
            vecAdd<<<dimGrid, dimBlock, 0, stream[i]>>>(deviceInput1 + i * inputLength / S_seg, deviceInput2 + i * inputLength / S_seg, deviceOutput + i * inputLength / S_seg, inputLength / S_seg);
        }
    }

    //@@ Copy the GPU memory back to the CPU here

    for (int i = 0; i < S_seg; i++) {
        if (i == S_seg - 1) {
            cudaMemcpyAsync(hostOutput + i * inputLength / S_seg, deviceOutput + i * inputLength / S_seg, (inputLength / S_seg + inputLength % S_seg) * sizeof(DataType), cudaMemcpyDeviceToHost, stream[i]);
        } else {
            cudaMemcpyAsync(hostOutput + i * inputLength / S_seg, deviceOutput + i * inputLength / S_seg, inputLength / S_seg * sizeof(DataType), cudaMemcpyDeviceToHost, stream[i]);
        }
    }

    //@@ Insert code below to synchronize streams
    cudaDeviceSynchronize();

    //@@ Insert code below to compare the output with the reference

    int errors = 0;
    for (int i = 0; i < inputLength; i++) {
        if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
            errors++;
        }
    }

    std::cout << "Error: " << errors << std::endl;

    // print out the first 10 differences
    for (int i = 0; i < inputLength; i++) {
        std::cout << "Host output: " << hostOutput[i] << " Reference: " << resultRef[i] <<  " Diff: " << hostOutput[i] - resultRef[i] << std::endl;
    }

    //@@ Destroy cuda streams here
    for (int i = 0; i < S_seg; i++) {
        cudaStreamDestroy(stream[i]);
    }

    //@@ Free the GPU memory here

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here

    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);

    return 0;
}
