#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "cblas.h"
#include "gemm.hpp"

extern "C" {
void bl_dgemm(int m, int n, int k, double *A, int lda, double *B, int ldb,
              double *C, int ldc);
extern void dgemm_(char *, char *, int *, int *, int *, double *, double *,
                   int *, double *, int *, double *, double *, int *);
}

class Time {
private:
    uint64_t mStart, mEnd;

public:
    uint64_t start() {
        mStart = now();
        return mStart;
    }
    uint64_t end() {
        mEnd = now();
        return mEnd;
    }
    double s() { return static_cast<double>(mEnd - mStart) / 1000000.0; }
    uint64_t now() {
        timeval _now;
        gettimeofday(&_now, static_cast<struct timezone *>(0));
        return static_cast<uint64_t>(_now.tv_sec) * 1000000 + _now.tv_usec;
    }
};

class Log {
public:
    Log() {
        fp = fopen("gemm.info.txt", "a");
        fprintf(fp, "\n %lud\n", Time().start());
        fprintf(fp, "| %12s | %6s | %7s | %9s | %24s | %7s |\n", "name", "N",
                "time", "gflops", "C[-1]", "norm");
        printf("| %12s | %6s | %7s | %9s | %24s | %7s |\n", "name", "N", "time",
               "gflops", "C[-1]", "norm");
    }
    ~Log() { fclose(fp); }
    void info(const char *name, double tm, double gflops, int n, double last,
              double norm) {
        printf("| %12s | %6d | %4.5f | %4.5f | %20.5f | %3.5f |\n", name, n, tm,
               gflops, last, norm);
        fprintf(fp, "| %12s | %6d | %4.5f | %4.5f | %20.5f | %3.5f |\n", name,
                n, tm, gflops, last, norm);
    }

private:
    FILE *fp;
};
#define TIMING(msg, x)                                                   \
    do {                                                                 \
        fill(A);                                                         \
        fill(B);                                                         \
        resetC(C);                                                       \
        Time now;                                                        \
        now.start();                                                     \
        x;                                                               \
        now.end();                                                       \
        double diff = simple::norm(C.data(), refC.data(), N * N);        \
        log.info(msg, now.s(), gflops / now.s(), N, C[N * N - 1], diff); \
    } while (0);
#define TIMING000(msg, x)                                                   \
    do {                                                                    \
        fill(A);                                                            \
        fill(B);                                                            \
        resetC(refC);                                                       \
        Time now;                                                           \
        now.start();                                                        \
        x;                                                                  \
        now.end();                                                          \
        double diff = simple::norm(refC.data(), refC.data(), N * N);        \
        log.info(msg, now.s(), gflops / now.s(), N, refC[N * N - 1], diff); \
    } while (0);

#define LOOP for (int loopi = 0; loopi < LOOPS; loopi++)

namespace simple {

template <typename T>
double norm(T *A, T *B, size_t n) {
    double thenorm = 0.0;
#pragma omp parallel for reduction(+ : thenorm)
    for (size_t i = 0; i < n; i++) {
        T diff = A[i] - B[i];
        double d = static_cast<double>(diff);
        thenorm += d * d;
    }
    return std::sqrt(thenorm);
}

}  // namespace simple
// C[0,0] = alpha * A[0,x] * B[x,0] + beta * C[0,0]
template <typename T>
double dgemm_one(int K, T alpha, T *A, int lda, T *B, int ldb, T beta, T *C) {
    double sum = 0;
    for (int k = 0; k < K; k++) {
        sum += beta * A[k] * B[k * ldb];
    }
    sum += C[0];
    return sum;
}

void resetC(MM::simple::vector<double> &A) {
    for (size_t i = 0; i < A.size(); i++) A[i] = 1.0;
}
void fill(MM::simple::vector<double> &A) {
    for (size_t i = 0; i < A.size(); i++) A[i] = static_cast<double>(i);
}
void help() {
    printf("%s",
           " -n n : matrix size M=N=K=n\n"
           " -l l : loop the dgemm l times\n"
           " -K k : matrix size M=N=K=1024*k\n"
           " -t t : number of threads\n");
}
typedef double T;
#ifndef NUM_CPU_CORES
#define NUM_CPU_CORES 1
#endif
int main(int argc, char **argv) {
    int N = 4096;
    int LOOPS = 1;
    int opt;
    int b = 256;
    int threads = NUM_CPU_CORES;
    (void)threads;
    Log log;
    while ((opt = getopt(argc, argv, "n:l:K:t:b:")) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                break;
            case 'l':
                LOOPS = atoi(optarg);
                break;
            case 'K':
                N = 1024 * atoi(optarg);
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 't':
                threads = atoi(optarg);
                MM::simple::setNumThreads(atoi(optarg));
                break;
            default: /* '?' */
                help();
                exit(-1);
                break;
        }
    };

    int len = N * N;
    double gflops = 2.0 * N * N * N * 1.0e-09;
    double alpha = 1.0, beta = 1.0;

    MM::simple::vector<double> A(len);
    MM::simple::vector<double> B(len);
    MM::simple::vector<double> C(len, beta);
    MM::simple::vector<double> refC(len, beta);
    fill(A);
    fill(B);
    // the refernce refnn of refC[N-1,N-1]
    double refnn =
        dgemm_one(N, alpha, &A[len - N], N, &B[N - 1], N, beta, &C[len - 1]);
    // clang-format off
    LOOP TIMING000("cblas_dgemm", cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, refC, N));
    //LOOP TIMING("dgemm_", dgemm_("N", "N", &N, &N, &N, &alpha, A.data(), &N, B.data(), &N, &beta, C.data(), &N));
#if defined(ENABLE_BLISLAB) && ENABLE_BLISLAB == 1
    LOOP  TIMING("blislab_gemm", bl_dgemm(N, N, N, A, N, B, N, C, N));
#endif
    //LOOP  TIMING("gemm", (MM::gemm<T, 128, 128, 128, 4>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    //LOOP TIMING("gemm001", (MM::gemm001<T, 128, 256, 256, 4, 8>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    if( b == 0){
        LOOP TIMING("gemm002", (MM::gemm002<T, 128, 0, 256, 6>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    } else if( b == 256)     {
        LOOP TIMING("gemm002", (MM::gemm002<T, 128, 256, 256, 6>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    } else if( b == 320)    {
        LOOP TIMING("gemm002", (MM::gemm002<T, 128, 320, 256, 6>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    } else {
        //LOOP TIMING("gemm002", (MM::gemm002<T, 128, 256, 256, 6>(N, N, N, alpha, A, N, B, N, beta, C, N)));
        LOOP TIMING("gemm002", (MM::gemm002<T, 128, 512, 256, 6>(N, N, N, alpha, A, N, B, N, beta, C, N)));
    }
    fprintf(stderr, "ref[n-1,n-1]                                  | %f\n", refnn);
    // clang-format on
    return 0;
}
