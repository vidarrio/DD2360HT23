
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

    int S_seg;
    int segments;
    int streams = 4;

    //@@ Insert code below to read in inputLength from args
    if (argc != 3) {
        printf("Usage: vecAdd inputLength segmentSize\n");
        exit(1);
    }

    inputLength = atoi(argv[1]);
    S_seg = atoi(argv[2]);
    segments = ceil((double)inputLength / S_seg);

    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));

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

    //@@ Start cuda profiler
    cudaProfilerStart();

    //@@ Start cpu timer
    double cpu_start_time = get_wall_time();

    //@@ Insert code to create cuda streams here
    cudaStream_t stream[streams];

    for (int i = 0; i < streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    //@@ Initialize the 1D grid and block dimensions here
    // 1024 is the maximum number of threads per block for compute capability 6.1
    int threadsPerBlock = 1024;

    // round up to the nearest integer so we don't have a partial block
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock << std::endl;

    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    //@@ Insert code below to do the vector addition on the GPU
    for (int i = 0; i < segments; i++) {
        int j = i % streams;
        int cur_size = i * S_seg + S_seg > inputLength ? inputLength - i * S_seg : S_seg;
        
        //@@ Insert code to below to Copy memory to the GPU here
        cudaMemcpyAsync(deviceInput1 + i * S_seg, hostInput1 + i * S_seg, cur_size * sizeof(DataType), cudaMemcpyHostToDevice, stream[j]);
        cudaMemcpyAsync(deviceInput2 + i * S_seg, hostInput2 + i * S_seg, cur_size * sizeof(DataType), cudaMemcpyHostToDevice, stream[j]);
        
        //@@ Launch the GPU Kernel here
        vecAdd<<<dimGrid, dimBlock, 0, stream[j]>>>(deviceInput1 + i * S_seg, deviceInput2 + i * S_seg, deviceOutput + i * S_seg, cur_size);

        //@@ Copy the GPU memory back to the CPU here
        cudaMemcpyAsync(hostOutput + i * S_seg, deviceOutput + i * S_seg, cur_size * sizeof(DataType), cudaMemcpyDeviceToHost, stream[j]);
    }

    //@@ Insert code below to synchronize streams
    cudaDeviceSynchronize();

    //@@ Stop cpu timer
    double cpu_end_time = get_wall_time();

    //@@ Stop cuda profiler
    cudaProfilerStop();

    //@@ Insert code below to compare the output with the reference
    int errors = 0;
    for (int i = 0; i < inputLength; i++) {
        if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
            errors++;
        }
    }

    std::cout << "Error: " << errors << std::endl;

    //@@ Insert code below to print out timing information
    std::cout << "Execution time: " << cpu_end_time - cpu_start_time << std::endl;

    //@@ Destroy cuda streams here
    for (int i = 0; i < streams; i++) {
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
