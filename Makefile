all:
	nvcc -DBLOCK_SIZE=16 -o matmul matmul.cu
clean:
	rm -rf matmul