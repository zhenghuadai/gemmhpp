# gemmhpp
GEMM by C++ template

# BUILD
- git clone https://github.com/zhenghuadai/gemmhpp.git
- sudo apt install libopenblas-dev 
- cd gemmhpp
- mkdir build.rel
- cd build.rel
- cmake ..
- make
- ./gemm

# USAGE

- #include "gemm.hpp"
- // C = alhpa * A * B + beta * C
- MM::gemm002<T, MB, NB, KB, 6>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
