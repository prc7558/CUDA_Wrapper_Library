// header files
// standard headers
#include <windows.h>
#include <stdio.h>
#include <math.h>

// cuda headers
#include <cuda.h>

// global variables for device memory
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

// dll entry point function
BOOL WINAPI DllMain(HMODULE hDll, DWORD dwReason, LPVOID lpReserved)
{
    // code
    switch(dwReason)
    {
        case DLL_PROCESS_ATTACH: // used for initialization
            break;

        case DLL_THREAD_ATTACH: // used for initialization
            break;

        case DLL_THREAD_DETACH: // used for uninitialization
            break;

        case DLL_PROCESS_DETACH: // used for uninitialization
            break;

        default: 
            break;
    }

    return(TRUE);
}

// CUDA kernels
__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{
    // variable declarations
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // code
    if (i < len)
    {
        out[i] = in1[i] + in2[i];
    }
}

// definition of exportable function InitDeviceData()
extern "C" __declspec(dllexport) void InitDeviceData(float* hostInput1, float* hostInput2, int iNumberOfArrayElements)
{
    // variable declarations
    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result = cudaSuccess;

    // code
    // device memory allocation
    result = cudaMalloc((void**)&deviceInput1, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceInput1 array.\n");
        return;
    }

    result = cudaMalloc((void**)&deviceInput2, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceInput2 array.\n");
        if (deviceInput1)
        {
            cudaFree(deviceInput1);
            deviceInput1 = NULL;
        }
        return;
    }

    result = cudaMalloc((void**)&deviceOutput, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceOutput array.\n");
        if (deviceInput2)
        {
            cudaFree(deviceInput2);
            deviceInput2 = NULL;
        }
        if (deviceInput1)
        {
            cudaFree(deviceInput1);
            deviceInput1 = NULL;
        }
        return;
    }

    // copy data from host arrays into device arrays
    result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Host to Device Data Copy is failed for deviceInput1 array.\n");
        return;
    }

    result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Host to Device Data Copy is failed for deviceInput2 array.\n");
        return;
    }
}

// definition of exportable function RunVecAddKernel()
extern "C" __declspec(dllexport) void RunVecAddKernel(int iNumberOfArrayElements)
{
    // code
    if (deviceInput1 == NULL || deviceInput2 == NULL || deviceOutput == NULL)
    {
        printf("Device memory not initialized. Call InitDeviceData first.\n");
        return;
    }

    // CUDA kernel configuration
    dim3 dimGrid = dim3((int) ceil((float) iNumberOfArrayElements / 256.0f), 1, 1);
    dim3 dimBlock = dim3(256, 1, 1);

    // CUDA kernel for Vector Addition
    vecAddGPU <<<dimGrid, dimBlock >>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);
}

// definition of exportable function RetrieveDeviceDataAndCleanup()
extern "C" __declspec(dllexport) void RetrieveDeviceDataAndCleanup(float* hostOutput, int iNumberOfArrayElements)
{
    // variable declarations
    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result = cudaSuccess;

    // code
    if (deviceOutput != NULL)
    {
        // copy data from device array into host array
        result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            printf("Device to Host Data Copy is failed for hostOutput array.\n");
        }
    }

    // cleanup
    if (deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = NULL;
    }
    if (deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2 = NULL;
    }
    if (deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1 = NULL;
    }
}
