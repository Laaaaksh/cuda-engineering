NVCC := nvcc
ARCH_FLAGS := -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH_FLAGS)

TARGET := coalescing_demo.out
SRC := coalescing_demo.cu

$(TARGET): $(SRC) ../common/check.cuh
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

.PHONY: run clean
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
