# Generates the input files used by the pynvjitlink binding test suite

NVCC_FLAGS := -O3 -rdc true
CUBIN_FLAGS := -gencode arch=compute_75,code=sm_75 --cubin
FATBIN_FLAGS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 --fatbin
LTOIR_FLAGS := -gencode arch=lto_75,code=lto_75 -dc
PTX_FLAGS := -gencode arch=compute_75,code=compute_75 -ptx
OBJECT_FLAGS := -gencode arch=compute_75,code=compute_75 -dc
LIBRARY_FLAGS := -gencode arch=compute_75,code=compute_75 -lib


OUTPUT_DIR := ../pynvjitlink/tests

all:
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $(OUTPUT_DIR)/undefined_extern.cubin undefined_extern.cu
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.cubin test.cu
	nvcc $(NVCC_FLAGS) $(FATBIN_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.fatbin test.cu
	nvcc $(NVCC_FLAGS) $(LTOIR_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.ltoir test.cu
	nvcc $(NVCC_FLAGS) $(PTX_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.ptx test.cu
	nvcc $(NVCC_FLAGS) $(OBJECT_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.o test.cu
	nvcc $(NVCC_FLAGS) $(LIBRARY_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.a test.cu
