rm -rf test.qdstrm
rm -rf test.nsys-rep
rm -rf test.sqlite
rm -rf test
nvcc $1 -o test 
nsys profile --trace=cuda -o test --cudabacktrace true --force-overwrite true ./test #generate qdstrm file
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter test.qdstrm #generate nsys-rep file
# nsys analyze test.nsys-rep
nsys stats --report gpukernsum test.nsys-rep #generate sqlite file
rm -rf test.qdstrm
rm -rf test.nsys-rep
rm -rf test.sqlite
rm -rf test

# nsys stats --report
# nvtxsum
# osrtsum
# cudaapisum
# gpukernsum
# gpumemtimesum
# gpumemsizesum
# openmpevtsum
# khrdebugsum
# khrdebuggpusum
# vulkanmarkerssum
# vulkangpumarkersum
# dx11pixsum
# dx12gpumarkersum
# dx12pixsum
# wddmqueuesdetails
# unifiedmemory
# unifiedmemorytotals
# umcpupagefaults
# openaccsum