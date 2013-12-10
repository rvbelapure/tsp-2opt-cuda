CC = g++ 
NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -lm -DUNIX -O3 -m64 -fPIC -fno-strict-aliasing
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc -G -O3 -m64 --ptxas-options=-v -lcuda -lcuda -fmad=true -prec-div=false -ftz=true -prec-sqrt=false --use_fast_math
COPTFLAGS = -g
LDFLAGS =

all: tsp-cuda

tsp-cuda: tsp_cuda.o driver.o 
	$(CC) tsp_cuda.o driver.o  $(CFLAGS) $(COPTFLAGS) -o tsp-cuda

tsp_cuda.o: tsp_cuda.cu tsp_cuda.h tsp.h
	$(NVCC) -c tsp_cuda.cu $(COPTFLAGS) $(NVCCFLAGS)

driver.o: driver.cpp tsp.h
	$(CC) -c driver.cpp $(CFLAGS) $(COPTFLAGS)

clean:
	rm -f *.o dummy tsp-cuda prj-ga.* tour.dat tour.png
