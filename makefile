#CXX=clang++
#CC=clang
NUM_PROCESSORS := $(shell lscpu | awk '/^CPU\(s\):/ {print $$2}')
NUM_CORES := $(shell lscpu | awk -F':' '/^Core\(s\) per socket:/ {gsub (" ", "", $$2);print $$2}')
L1_SIZE := $(shell lscpu | awk -F':' '/^L1d cache:/ {gsub (" ", "", $$2);print $$2}')
L2_SIZE := $(shell lscpu | awk -F':' '/^L2 cache:/ {gsub (" ", "", $$2);print $$2}')
$(info L1_SIZE ${L1_SIZE})
$(info L2_SIZE ${L2_SIZE})
BLAS=-lblas
BLAS=-lopenblas
SRC=test/test.cpp

FLAGS := -DNUM_CPU_CORES=${NUM_CORES} -I./include
gemm:${SRC} 
	@echo -DNUM_CPU_CORES=${NUM_CORES}
	${CXX} -o gemm ${SRC}  -O3 ${BLAS}  -mavx2 -mfma -L. -fopenmp ${FLAGS} #-pg
gemm.s:${SRC} 
	${CXX} -o gemm.s -g -S ${SRC}  -O3 ${BLAS}  -mavx2 -mfma -L. -fopenmp ${FLAGS} #-pg
gemm.st:${SRC} gemm.hpp 
	${CXX} -o gemm.st ${SRC}  -O3 ${BLAS}  ${FLAGS} -mavx2 -mfma -L. -pg # -fopenmp 
gemm.sse:${SRC} 
	${CXX} -o gemm.sse ${SRC}  -O3 ${BLAS}  ${FLAGS} -fopenmp -msse3 -L. #-pg
gemm.0:${SRC} 
	${CXX} -o gemm.0 ${SRC}  -O3 ${BLAS}  ${FLAGS} -fopenmp  -L. #-pg
gemm.dbg:${SRC}
	${CXX} -o $@ $^ -O0 -lblas -mavx2 -mfma -g -L.  ${FLAGS}#-pg -fopenmp 
%.o:%.c
	#gcc -c -o $@ $^ -O3 -lblas -fopenmp -mavx -mfma
	${CC}  -O2 -Wall -msse3 -c $<  -o $@  ${FLAGS} -fopenmp
clean:
	rm -f gemm gemm.gdb *.o
gemm:include/gemm.hpp
gemm.sse:include/gemm.hpp
gemm.0:include/gemm.hpp
gemm.s:include/gemm.hpp
