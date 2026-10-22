#pragma once

// exported function declarations
extern "C" void InitDeviceData(float* hostInput1, float* hostInput2, int iNumberOfArrayElements);
extern "C" void RunVecAddKernel(int iNumberOfArrayElements);
extern "C" void RetrieveDeviceDataAndCleanup(float* hostOutput, int iNumberOfArrayElements);
