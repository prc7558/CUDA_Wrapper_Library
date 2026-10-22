// header files
// standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// custom headers
#include "helper_timer.h"
#include "CudaWrapper.h"

// global variables
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutputCPU = NULL;
float* hostOutputGPU = NULL;

int main(void)
{
    // function declarations
    void fillFloatArrayWithRandomNumbers(float*, int);
    void vecAddCPU(const float*, const float*, float*, int);
    void cleanup(void);

    // variable declarations
    const int iNumberOfArrayElements = 11444777;
    int size = iNumberOfArrayElements * sizeof(float);

    float timeOnCPU = 0.0f;
    float timeGPUInitAndTransfer = 0.0f;
    float timeGPUCompute = 0.0f;
    float timeGPURetrieveAndCleanup = 0.0f;

    StopWatchInterface* timer = NULL;

    // code
    // host memory allocation
    hostInput1 = (float*)malloc(size);
    if (hostInput1 == NULL)
    {
        printf("Host Memory allocation is failed for hostInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if (hostInput2 == NULL)
    {
        printf("Host Memory allocation is failed for hostInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutputCPU = (float*)malloc(size);
    if (hostOutputCPU == NULL)
    {
        printf("Host Memory allocation is failed for hostOutputCPU array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutputGPU = (float*)malloc(size);
    if (hostOutputGPU == NULL)
    {
        printf("Host Memory allocation is failed for hostOutputGPU array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into host arrays
    fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

    // vector addition on host
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    vecAddCPU(hostInput1, hostInput2, hostOutputCPU, iNumberOfArrayElements);

    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // vector addition on device (Init and Transfer)
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    InitDeviceData(hostInput1, hostInput2, iNumberOfArrayElements);

    sdkStopTimer(&timer);
    timeGPUInitAndTransfer = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // vector addition on device (Compute)
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    RunVecAddKernel(iNumberOfArrayElements);

    sdkStopTimer(&timer);
    timeGPUCompute = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // vector addition on device (Retrieve and Cleanup)
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    RetrieveDeviceDataAndCleanup(hostOutputGPU, iNumberOfArrayElements);

    sdkStopTimer(&timer);
    timeGPURetrieveAndCleanup = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // comparison
    const float epsilon = 0.000001f;
    int breakValue = -1;
    bool bAccuracy = true;
    for (int i = 0; i < iNumberOfArrayElements; i++)
    {
        float val1 = hostOutputCPU[i];
        float val2 = hostOutputGPU[i];
        if (fabs(val1 - val2) > epsilon)
        {
            bAccuracy = false;
            breakValue = i;
            break;
        }
    }

    char str[128];
    if (bAccuracy == false)
    {
        sprintf(str, "Comparison of CPU and GPU Vector Addition is not within accuracy of 0.000001 at array index %d", breakValue);
    }
    else
    {
        sprintf(str, "Comparison of CPU and GPU Vector Addition is within accuracy of 0.000001");
    }

    // output
    printf("Time taken for Vector Addition on CPU = %.6f\n", timeOnCPU);
    printf("Time taken by GPU for Memory/Transfer = %.6f\n", timeGPUInitAndTransfer);
    printf("Time taken by GPU for Pure Math = %.6f\n", timeGPUCompute);
    printf("Time taken by GPU for Retrieval/Cleanup = %.6f\n", timeGPURetrieveAndCleanup);
    printf("%s\n", str);

    // cleanup
    cleanup();

    return(0);
}

void fillFloatArrayWithRandomNumbers(float* arr, int len)
{
    // code
    const float fscale = 1.0f / (float) RAND_MAX;
    for (int i = 0; i < len; i++)
    {
        arr[i] = fscale * rand();
    }
}

void vecAddCPU(const float* arr1, const float* arr2, float *out, int len)
{
    // code
    for (int i = 0; i < len; i++)
    {
        out[i] = arr1[i] + arr2[i];
    }
}

void cleanup(void)
{
    // code
    if (hostOutputGPU)
    {
        free(hostOutputGPU);
        hostOutputGPU = NULL;
    }

    if (hostOutputCPU)
    {
        free(hostOutputCPU);
        hostOutputCPU = NULL;
    }

    if (hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}
