NVCC := nvcc
# Covers Volta through Ada so the same binary runs on most GPUs you'd
# actually learn on: T4 (Colab free tier, sm_75), V100 (sm_70), A100
# (sm_80), RTX 30-series (sm_86), RTX 40-series (sm_89).
ARCH_FLAGS := -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH_FLAGS)

TARGET := hello_kernel.out
SRC := hello_kernel.cu

$(TARGET): $(SRC) ../common/check.cuh
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

.PHONY: run clean
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
