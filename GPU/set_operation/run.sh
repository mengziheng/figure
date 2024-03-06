exe=$1
nvcc $exe -o test && ./test
rm -rf test