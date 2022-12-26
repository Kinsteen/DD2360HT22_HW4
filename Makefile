ARCH = sm_86
NVCC_OPTIONS = -g -std=c++14 -arch=$(ARCH)

all: lab4_ex2 lab4_ex3 lab4_ex4

lab4_ex2: lab4_ex2.cu
	nvcc $(NVCC_OPTIONS) $< -o $@

lab4_ex3: lab4_ex3.cu
	nvcc $(NVCC_OPTIONS) $< -o $@

lab4_ex4: lab4_ex4.cu
	nvcc $(NVCC_OPTIONS) -lcusparse -lcublas $< -o $@
test: test.cu
	nvcc $(NVCC_OPTIONS) -lcusparse -lcublas $< -o $@

clean:
	rm lab4_ex2
	rm lab4_ex3
	rm lab4_ex4
.PHONY: clean