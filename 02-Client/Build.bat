@echo off

echo "================== compiling =================="
cl.exe /c /EHsc DemoClient.cpp

echo "================== linking =================="
link.exe DemoClient.obj CudaWrapper.lib /SUBSYSTEM:CONSOLE
