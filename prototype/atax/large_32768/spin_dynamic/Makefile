
all: atax

atax: atax.cu
	nvcc -Xcompiler -fopenmp -lcuda -lcudart -lgomp   -o atax atax.cu

clean:
	rm -rf atax
