#include <immintrin.h>
#include <omp.h>
#ifdef __AVX__
typedef __m256d __d4v;
inline __d4v _avx0() { return _mm256_setzero_pd(); }
inline __d4v _avx_broadcast(double const *__X) {
    return _mm256_broadcast_sd(__X);
}
__inline __d4v _avx_load_d4(double const *__P) { return _mm256_load_pd(__P); }
#elif defined __SSE3__
class __d4v {
public:
    __m128d d[2];
    __d4v() {}
    __d4v(const __m128d &o0, const __m128d &o1) {
        d[0] = o0;
        d[1] = o1;
    }
    __d4v(const __d4v &o) {
        d[0] = o.d[0];
        d[1] = o.d[1];
    }
    __d4v &operator+=(const __d4v &s) {
        this->d[0] += s.d[0];
        this->d[1] += s.d[1];
        return *this;
    }
    __d4v &operator*=(const __d4v &s) {
        this->d[0] *= s.d[0];
        this->d[1] *= s.d[1];
        return *this;
    }

    __d4v &operator*=(const double &s1) {
        this->d[0] *= s1;
        this->d[1] *= s1;
        return *this;
    }
    __d4v operator*(const __d4v &s1) {
        __d4v d;
        d.d[0] = this->d[0] * s1.d[0];
        d.d[1] = this->d[1] * s1.d[1];
        return d;
    }
};
// typedef __m128d __d4v[2];
inline __d4v _avx0() { return __d4v(_mm_setzero_pd(), _mm_setzero_pd()); }
inline __d4v _avx_broadcast(double const *__X) {
    __m128d x = _mm_loaddup_pd(__X);
    return __d4v(x, x);
}
__inline __d4v _avx_load_d4(double const *__P) {
    return __d4v{_mm_load_pd(__P), _mm_load_pd(__P + 2)};
}

#else
typedef __m256d __d4v;
inline __d4v _avx0() { return __d4v{0.0, 0.0, 0.0, 0.0}; }
inline __d4v _avx_broadcast(double const *__X) {
    return __d4v{*__X, *__X, *__X, *__X};
}
__inline __d4v _avx_load_d4(double const *__P) { return *(__d4v *)__P; }
#endif

#include <memory>

#define MATRIX_ROW_MAJOR 1
#ifdef MATRIX_ROW_MAJOR
#define ij(Matrix, stride, i, j) Matrix[(i) * (stride) + j]
#define tij(Matrix, stride, i, j) Matrix[(j) * (stride) + i]
#else
#define ij(Matrix, stride, i, j) Matrix[(j) * (stride) + i]
#define tij(Matrix, stride, i, j) Matrix[(i) * (stride) + j]
#endif

#define _A(i, j) ij(A, lda, i, j)
#define _B(i, j) ij(B, ldb, i, j)
#define _C(i, j) ij(C, ldc, i, j)

#define _bA(bi, bj, i, j) _A((bi) + i, (bj) + j)
#define _bB(bi, bj, i, j) _B((bi) + i, (bj) + j)
#define _bC(bi, bj, i, j) _C((bi) + i, (bj) + j)

#define _tA(bi, bj, i, j) tij(A, lda, (bi) + i, (bj) + j)
#define _tB(bi, bj, i, j) tij(B, ldb, (bi) + i, (bj) + j)
#define _tC(bi, bj, i, j) tij(C, ldc, (bi) + i, (bj) + j)

#define _cA(bi, bj, i, j) pblkA[(i)*KB + j]
#define _cB(bi, bj, i, j) pblkB[(i)*NB + j]
#define _cC(bi, bj, i, j) pblkC[(i)*NB + j]

#define _c4B(bk, bj, k, j) pblkB[(j)*KB + (k)*4]
#define _c4A(bi, bk, i, k) pblkA[(k)*4 + (i)]
#define _c6A(bi, bk, i, k) pblkA[(k)*6 + (i)]
#define _c8A(bi, bk, i, k) pblkA[(k)*8 + (i)]
#define _cXA(bi, bk, i, k, Stride) pblkA[(k)*(Stride) + (i)]
#define _c4AAAA(bi, bk, i, k) pblkA[((k) >> 2) * 16 + (i)*4 + ((k)&3)]

#define _c8B(bk, bj, k, j, p) pblkB[(j)*KB + (k)*8 + (p)]
#define _c12B(bk, bj, k, j, p) pblkB[(j)*KB + (k)*12 + (p)]
#define _cXB(bk, bj, k, j, p, Stride) pblkB[(j)*KB + (k)*Stride + (p)]
template <uint32_t i, uint32_t N, uint32_t step = 1>
struct unroll {
    template <typename F>
    inline static void call(F const &f) {
        f(i);
        unroll<i + step, N, step>::call(f);
    }
};

template <uint32_t i, uint32_t step>
struct unroll<i, i, step> {
    template <typename F>
    inline static void call(F const &f) {}
};

template <uint32_t i>
struct llornu {
    template <typename F>
    inline static void call(F const &f) {
        f(i - 1);
        llornu<i - 1>::call(f);
    }
};
template <>
struct llornu<0u> {
    template <typename F>
    inline static void call(F const &) {}
};
// llornu<RMB>::call( [&] (uint32_t x){dd0[x] = _avx0() ;});

// clang-format off
#define FOR(x, X, oneline)\
    for (int x = 0; x < X; x++) { oneline; }
#define FORX(x, X, step, oneline)\
    for (int x = 0; x < X; x+=step) { oneline; }
#define FORXX(x, X, oneline, signal)\
    do{\
        for (int x = 0; x < X-1; x++) { oneline; } \
        signal;                                    \
        {int x = X-1; oneline;}                    \
    }while(0);
#define FORXX2(x, X, oneline, signal)\
    do{\
        for (int x = 0; x < X-2; ) { {oneline;x++;} {oneline; x++;}  } \
        signal;                                    \
        {int x = X-2; oneline; x++; oneline;}                    \
    }while(0);
#define FORXX4(x, X, oneline, signal)\
    do{\
        for (int x = 0; x < X-4; ) {  {oneline;x++;} {oneline; x++;} {oneline;x++;} {oneline; x++;}  } \
        signal;                                    \
        {int x = X-4; oneline; x++; oneline; x++; oneline; x++; oneline;}                    \
    }while(0);
#define IN ,
// clang-format on
#if 1
#define UNROLL(x, X, oneline) \
    _Pragma("GCC unroll 1") for (int x = 0; x < X; x++) { oneline; }
#define UNROLLX(x, X, step, oneline) \
    _Pragma("GCC unroll 1") for (int x = 0; x < X; x += step) { oneline; }
#else
#define UNROLL(x, X, oneline) \
    unroll<0, X, 1>::call([&](uint32_t x) { oneline; });
#define UNROLLX(x, X, step, oneline) \
    unroll<0, X, step>::call([&](uint32_t x) { oneline; });
#endif
namespace MM {
namespace simple {
uint32_t gNumThreads = NUM_CPU_CORES;
void setNumThreads(uint32_t n) { gNumThreads = n; }
uint32_t getNumThreads() { return gNumThreads; }
template <typename T>
class vector {
public:
    typedef std::unique_ptr<vector<T>> Ptr;
    vector(size_t n) : m_n(n) { allocate(); }
    vector(size_t n, T t) : m_n(n) {
        allocate();
        fill(t);
    }
    vector(vector &&o) : m_n(o.m_n), m_data(o.m_data), m_rsc(o.m_rsc) {
        m_n = 0;
        m_data = nullptr;
        m_rsc = nullptr;
    }
    ~vector() {
        if (m_rsc) free(m_rsc);
        m_rsc = nullptr;
        m_data = nullptr;
    }

    void fill(T t) {
        for (size_t i = 0; i < m_n; i++) m_data[i] = t;
    }

    size_t size() { return m_n; }
    T *data() { return m_data; }
    T &operator[](int i) { return m_data[i]; }
    operator const T *() const { return m_data; }
    operator T *() { return m_data; }

private:
    void allocate() {
        m_rsc = (T *)malloc(m_n * sizeof(T) + 256 / sizeof(T));
        m_data = align(m_rsc);
    }

    uintptr_t ptou(T *p) { return reinterpret_cast<std::uintptr_t>(p); }
    T *utop(uintptr_t p) { return reinterpret_cast<T *>(p); }
    T *align(T *a) { return utop((ptou(a) + 255) & (~0xffuLL)); }

public:
    size_t m_n;
    T *m_data;
    T *m_rsc;
};
}  // namespace simple
////////////////////////////////////////////
//
//   -----------         00001111222233334444
//   -----------         xxxxyyyyzzzzwwwwtttt
//   --0000xxxx-
//   --1111yyyy-
//   --2222zzzz-    -->
//   --3333wwww-
//   --4444tttt-
//   -----------
//   -----------
//
////////////////////////////////////////////
void pack4B(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    __d4v *p = (__d4v *)dst;
    for (int j = 0; j < cols; j += 4) {
        for (int i = 0; i < rows; i++) {
            *p = *(__d4v *)&src[i * lds + j];
            p++;
        }
    }
}

////////////////////////////////////////////
//
//   -----------         00001111222233334444555566667777....
//   -----------         xxxxyyyyzzzzwwwwrrrrssssttttqqqq....
//   --01234567-
//   --01234567-
//   --01234567-    -->
//   --01234567-
//   --xyzwrstq-
//   --xyzwrstq-
//   --xyzwrstq-
//   --xyzwrstq-
//
////////////////////////////////////////////
#if !defined(__AVX__)
void pack4A(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    double *p = dst;
    for (int i = 0; i < rows; i += 4) {
        for (int j = 0; j < cols; j++) {
            p[0] = src[(i + 0) * lds + j];
            p[1] = src[(i + 1) * lds + j];
            p[2] = src[(i + 2) * lds + j];
            p[3] = src[(i + 3) * lds + j];
            p += 4;
        }
    }
}
#else
void pack4A(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    __d4v *p = (__d4v *)dst;
    // for (int i = 0; i < rows; i += 4)
    int i = 0;
    {
        for (int j = 0; j < cols; j += 4) {
            __m256d p0 = *(__d4v *)&src[(i + 0) * lds + j];
            __m256d p1 = *(__d4v *)&src[(i + 1) * lds + j];
            __m256d p2 = *(__d4v *)&src[(i + 2) * lds + j];
            __m256d p3 = *(__d4v *)&src[(i + 3) * lds + j];
            __m256d pab02 = _mm256_shuffle_pd(p0, p1, 0);
            __m256d pab13 = _mm256_shuffle_pd(p0, p1, 0xf);
            __m256d pcd02 = _mm256_shuffle_pd(p2, p3, 0);
            __m256d pcd13 = _mm256_shuffle_pd(p2, p3, 0xf);
            p[0] =
                _mm256_insertf128_pd(pab02, _mm256_extractf128_pd(pcd02, 0), 1);
            p[1] =
                _mm256_insertf128_pd(pab13, _mm256_extractf128_pd(pcd13, 0), 1);
            p[2] =
                _mm256_insertf128_pd(pcd02, _mm256_extractf128_pd(pab02, 1), 0);
            p[3] =
                _mm256_insertf128_pd(pcd13, _mm256_extractf128_pd(pab13, 1), 0);
            p += 4;
        }
    }
}
#endif
void pack4AAAA(const double *src, int lds, double *dst, int /*ldd*/, int rows,
               int cols) {
    __d4v *p = (__d4v *)dst;
    // for (int i = 0; i < rows; i += 4)
    int i = 0;
    {
        for (int j = 0; j < cols; j += 4) {
            p[0] = *(__d4v *)&src[(i + 0) * lds + j];
            p[1] = *(__d4v *)&src[(i + 1) * lds + j];
            p[2] = *(__d4v *)&src[(i + 2) * lds + j];
            p[3] = *(__d4v *)&src[(i + 3) * lds + j];
            p += 4;
        }
    }
}

void pack6A(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    double *p = dst;
    for (int i = 0; i < rows; i += 6) {
        for (int j = 0; j < cols; j++) {
            p[j * 6 + 0] = src[(i + 0) * lds + j];
            p[j * 6 + 1] = src[(i + 1) * lds + j];
            p[j * 6 + 2] = src[(i + 2) * lds + j];
            p[j * 6 + 3] = src[(i + 3) * lds + j];
            p[j * 6 + 4] = src[(i + 4) * lds + j];
            p[j * 6 + 5] = src[(i + 5) * lds + j];
        }
    }
}

template <int ROWS>
void packA(const double *src, int lds, double *dst, int /*ldd*/, int rows,
           int cols) {
    double *p = dst;
    for (int i = 0; i < rows; i += ROWS) {
        for (int j = 0; j < cols; j++) {
            UNROLL(x, ROWS, p[j * ROWS + x] = src[(i + x) * lds + j]);
        }
    }
}

template <int ROWS>
void packA_(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    double *p = dst;
    for (int i = 0; i < rows; i += 1) {
        for (int j = 0; j < cols; j++) {
            p[j * ROWS + i] = src[(i + 0) * lds + j];
        }
    }
}
////////////////////////////////////////////
//
//   -----------
//   0000xxx1111yyyy2222zzzz3333wwww4444rrrr5555ssss6666tttt7777qqqq....
//   -----------
//   --01234567-
//   --01234567-
//   --01234567-    -->
//   --01234567-
//   --xyzwrstq-
//   --xyzwrstq-
//   --xyzwrstq-
//   --xyzwrstq-
//
////////////////////////////////////////////
void pack8A(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    double *p = dst;
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j++) {
            p[0] = src[(i + 0) * lds + j];
            p[1] = src[(i + 1) * lds + j];
            p[2] = src[(i + 2) * lds + j];
            p[3] = src[(i + 3) * lds + j];
            p[4] = src[(i + 4) * lds + j];
            p[5] = src[(i + 5) * lds + j];
            p[6] = src[(i + 6) * lds + j];
            p[7] = src[(i + 7) * lds + j];
            p += 8;
        }
    }
}

////////////////////////////////////////////
//
//   -----------         0000xxxx1111yyyy2222zzzz3333wwww4444tttt
//   -----------
//   --0000xxxx-
//   --1111yyyy-
//   --2222zzzz-    -->
//   --3333wwww-
//   --4444tttt-
//   -----------
//   -----------
//
////////////////////////////////////////////
void pack8B(const double *src, int lds, double *dst, int /*ldd*/, int rows,
            int cols) {
    __d4v *p = (__d4v *)dst;
    for (int j = 0; j < cols; j += 8) {
        for (int i = 0; i < rows; i++) {
            *p++ = *(__d4v *)&src[i * lds + j];
            *p++ = *(__d4v *)&src[i * lds + j + 4];
        }
    }
}

template <int COLS>
void packB(const double *src, int lds, double *dst, int /*ldd*/, int rows,
           int cols) {
    __d4v *p = (__d4v *)dst;
    for (int j = 0; j < cols; j += COLS) {
        for (int i = 0; i < rows; i++) {
            for (int v = 0; v < COLS; v += 4) {
                *p++ = *(__d4v *)&src[i * lds + j + v];
            }
        }
    }
}
void pack16B(const double *src, int lds, double *dst, int /*ldd*/, int rows,
             int cols) {
    __d4v *p = (__d4v *)dst;
    for (int j = 0; j < cols; j += 16) {
        for (int i = 0; i < rows; i++) {
            *p++ = *(__d4v *)&src[i * lds + j];
            *p++ = *(__d4v *)&src[i * lds + j + 4];
            *p++ = *(__d4v *)&src[i * lds + j + 8];
            *p++ = *(__d4v *)&src[i * lds + j + 12];
        }
    }
}

template <typename T, int MB, int NB, int KB, int RMB = 4>
void gemm(const int M, const int N, const int K, const T alpha, const T *A,
          const int lda, const T *B, const int ldb, const T beta, T *C,
          const int ldc) {
    typename simple::vector<T>::Ptr blkA;
    typename simple::vector<T>::Ptr blkB;
    typename simple::vector<T>::Ptr blkC;
    T *pblkA = nullptr;
    T *pblkB = nullptr;
    T *pblkC = nullptr;
    __d4v alpha4 = _avx_broadcast(&alpha);
#pragma omp parallel private(pblkA, pblkB, pblkC, blkA, blkB, \
                             blkC)  // num_threads(8)
    {
        // if (pblkA == nullptr)
        {
            blkA.reset(new simple::vector<T>(8 * KB + 1024));
            blkB.reset(new simple::vector<T>(KB * NB + 1024));
            blkC.reset(new simple::vector<T>(MB * NB + 1024));
            pblkA = (blkA->data());
            pblkB = (blkB->data());
            pblkC = (blkC->data());
            (void)pblkC;
        }

        for (int bi = 0; bi < M; bi += MB) {
#pragma omp for
            for (int bj = 0; bj < N; bj += NB) {
                // do C=beta*C on BlockC[bi, bj]
                for (int i = 0; i < MB; i++) {
                    for (int j = 0; j < NB; j += 4) {
                        __d4v *cdd0 = (__d4v *)(&_bC(bi, bj, i, j));
                        cdd0[0] *= beta;
                    }
                }
                for (int bk = 0; bk < K; bk += KB) {
                    // BlockC[bi, bj] += alpha*BlockA[bi, bk] * BlockB[bk, bj]
                    // clang-format off
                    int _MB = std::min(M - bi, MB);
                    int _NB = std::min(N - bj, NB);
                    int _KB = std::min(K - bk, KB);
                    packB<8>(&_bB(bk, bj, 0, 0), N, &pblkB[0*KB], 8, KB, NB);// _c8B(bk,bj,k,j)
                    for (int i = 0; i < _MB; i += RMB) {
                        pack4A(&_bA(bi, bk, i, 0), K, &pblkA[0], 4, 4, KB); // _c4A(bi,bk,i,k)
                        for (int j = 0; j < _NB; j += 8) {
                            _mm_prefetch(&_c8B(bk, bj, 0, j, 0),  _MM_HINT_T0 );
                            __d4v dd0[RMB], dd1[RMB];
                            UNROLL(x, RMB, dd0[x] = _avx0());
                            UNROLL(x, RMB, dd1[x] = _avx0());
                            for (int k = 0; k < _KB; k++) {
                                __d4v bb0 = _avx_load_d4(&_c8B(bk, bj, k, j, 0));
                                __d4v bb1 = _avx_load_d4(&_c8B(bk, bj, k, j, 4));
                                __d4v aa[RMB];
                                UNROLL(x, RMB, aa[x] = _avx_broadcast(&_c4A(bi, bk, /*i +*/ x, k)));
                                UNROLL(x, RMB, dd0[x] += bb0*aa[x]);
                                UNROLL(x, RMB, dd1[x] += bb1*aa[x]);
                            }

                            {
                                for(int x=0; x<RMB; x++) {
                                    *(__d4v *)(&_bC(bi, bj, i + x, j)) += dd0[x] * alpha4;
                                    *(__d4v *)(&_bC(bi, bj, i + x, j+4)) += dd1[x] * alpha4;
                                };
                            }
                        }
                    }
                    // clang-format on
                }
            }
        }
    }
}

template <typename T, int MB, int NB, int KB, int RMB = 4, int RNB = 8>
void gemm001(const int M, const int N, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
    typename simple::vector<T>::Ptr blkA;
    typename simple::vector<T>::Ptr blkB;
    typename simple::vector<T>::Ptr blkC;
    T *pblkA = nullptr;
    T *pblkB = nullptr;
    T *pblkC = nullptr;
    __d4v alpha4 = _avx_broadcast(&alpha);
#pragma omp parallel private(pblkA, pblkB, pblkC, blkA, blkB, blkC) \
    num_threads(simple::getNumThreads())
    {
        // if (pblkA == nullptr)
        {
            blkA.reset(new simple::vector<T>(8 * KB + 1024));
            blkB.reset(new simple::vector<T>(KB * NB * 2 + 1024));
            // blkC.reset(new simple::vector<T>(MB * NB + 1024));
            pblkA = (blkA->data());
            pblkB = (blkB->data());
            // pblkC = (blkC->data());
            (void)pblkC;
        }

        for (int bk = 0; bk < K; bk += KB) {
#pragma omp for
            for (int bj = 0; bj < N; bj += NB) {
                // clang-format off
                packB<RNB>(&_bB(bk, bj, 0, 0), N, &pblkB[0 * KB], RNB, KB, NB);  // _c8B(bk,bj,k,j)
                // clang-format on
                for (int bi = 0; bi < M; bi += MB) {
                    // do C=beta*C on BlockC[bi, bj]
                    if (bk == 0) {
                        for (int i = 0; i < MB; i++) {
                            for (int j = 0; j < NB; j += 4) {
                                __d4v *cdd0 = (__d4v *)(&_bC(bi, bj, i, j));
                                cdd0[0] *= beta;
                            }
                        }
                    }
                    // BlockC[bi, bj] += alpha*BlockA[bi, bk] * BlockB[bk, bj]
                    // clang-format off
                    int _MB = std::min(M - bi, MB);
                    int _NB = std::min(N - bj, NB);
                    int _KB = std::min(K - bk, KB);
                    for (int i = 0; i < _MB; i += RMB) {
                        pack4A(&_bA(bi, bk, i, 0), K, &pblkA[0], 4, 4, KB); // _c4A(bi,bk,i,k)
                        for (int j = 0; j < _NB; j += RNB) {
                            _mm_prefetch(&_cXB(bk, bj, 0, j, 0, RNB),  _MM_HINT_T0 );
                            __d4v dds[RMB][RNB/4];
                            UNROLL(y, RNB/4, ({UNROLL(x, RMB, dds[x][y] = _avx0());}));
                            FORXX(k, _KB,({
                                __d4v aa[RMB];
                                __d4v bbs[RNB/4];
                                UNROLLX(xx, RNB/4, 1, ({ bbs[xx] = _avx_load_d4(&_cXB(bk, bj, k, j, xx*4, RNB));}));
                                UNROLLX(x, RMB, 2, ({
                                            aa[x] = _avx_broadcast(&_c4A(bi, bk, /*i +*/ x, k));
                                            aa[x+1] = _avx_broadcast(&_c4A(bi, bk, /*i +*/ x+1, k));
                                            UNROLL(y, RNB/4, ({ dds[x][y] += bbs[y]*aa[x];}));
                                            UNROLL(y, RNB/4, ({ dds[x+1][y] += bbs[y]*aa[x+1];}));
                                            }));
                            }), _mm_prefetch(&_bC(bi, bj, i, j),  _MM_HINT_T0 ));

                            {
                                for(int x=0; x<RMB; x++) {
                                    UNROLL(y, RNB/4, ({ *(__d4v *)(&_bC(bi, bj, i + x, j + y*4)) += dds[x][y] * alpha4;}));
                                };
                            }
                        }
                    }
                    // clang-format on
                }
            }
        }
    }
}

template <typename T>
void betaC(int MB, int NB, T beta, T *C, int ldc) {
    if (beta == 1.0) return;
    for (int i = 0; i < MB; i++) {
        for (int j = 0; j < NB; j += 4) {
            __d4v *cdd0 = (__d4v *)(&_C(i, j));
            cdd0[0] *= beta;
        }
    }
}
#if 1
#define PREFETCH(...) _mm_prefetch(__VA_ARGS__)
#else
#define PREFETCH(...) 
#endif
template <typename T, int MB, int NB, int KB, int RMB = 6>
void gemm002(const int M, const int N, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
    typename simple::vector<T>::Ptr blkA;
    typename simple::vector<T>::Ptr blkB;
    typename simple::vector<T>::Ptr blkC;
    T *pblkA = nullptr;
    T *pblkB = nullptr;
    T *pblkC = nullptr;
    __d4v alpha4 = _avx_broadcast(&alpha);
    // clang-format off
    const int nNB =
        (NB > 0)
            ? NB
            : (((N / simple::getNumThreads() / ((N + 2047) / 2048) + 7) / 8) * 8);
    // clang-format on

#pragma omp parallel private(pblkA, pblkB, pblkC, blkA, blkB, blkC) \
    num_threads(simple::getNumThreads())
    {
        // if (pblkA == nullptr)
        {
            blkA.reset(new simple::vector<T>(8 * KB + 1024));
            blkB.reset(new simple::vector<T>(KB * nNB + 1024));
            // blkC.reset(new simple::vector<T>(MB * nNB + 1024));
            pblkA = (blkA->data());
            pblkB = (blkB->data());
            // pblkC = (blkC->data());
            (void)pblkC;
        }

        for (int bk = 0; bk < K; bk += KB) {
#pragma omp for
            for (int bj = 0; bj < N; bj += nNB) {
                // clang-format off
                packB<8>(&_bB(bk, bj, 0, 0), N, &pblkB[0 * KB], 8, KB, nNB);  // _c8B(bk,bj,k,j)
                // clang-format on
                for (int bi = 0; bi < M; bi += MB) {
                    if (bk == 0) {
                        // do C=beta*C on BlockC[bi, bj]
                        betaC(MB, nNB, beta, &_C(bi, bj), ldc);
                    }
                    // BlockC[bi, bj] += alpha*BlockA[bi, bk] * BlockB[bk, bj]
                    // clang-format off
                    int _MB = std::min(M - bi, MB);
                    int _NB = std::min(N - bj, nNB);
                    int _KB = std::min(K - bk, KB);
                    for (int i = 0; i < _MB; i += RMB) {
                        int _RMB = std::min(_MB - i, RMB);
                        if(_RMB == RMB){
                            packA<RMB>(&_bA(bi, bk, i, 0), K, &pblkA[0], _RMB, _RMB, KB); // _c4AAAA(bi,bk,i,k)
                            for (int j = 0; j < _NB; j += 8) {
                                __d4v dd0[RMB], dd1[RMB];
                                PREFETCH(&_c8B(bk, bj, 0, j, 0),  _MM_HINT_T1 );
                                PREFETCH(&_c8B(bk, bj, 0, j, 4),  _MM_HINT_T1 );
                                UNROLL(x, RMB, ({dd0[x] = _avx0(); dd1[x] = _avx0();}));
                                FORXX(k, _KB,({
                                    __d4v aa[RMB];
                                    __d4v bb0 = _avx_load_d4(&_c8B(bk, bj, k, j, 0));
                                    __d4v bb1 = _avx_load_d4(&_c8B(bk, bj, k, j, 4));
                                    if(j == 0)
                                    PREFETCH(&_cXA(bi, bk, 0, k, 0),  _MM_HINT_T0 );
                                    PREFETCH(&_c8B(bk, bj, k+1, j, 0),  _MM_HINT_T1 );
                                    PREFETCH(&_c8B(bk, bj, k+1, j, 4),  _MM_HINT_T1 );
                                    UNROLLX(x, RMB, 2, ({ 
                                                aa[x]     = _avx_broadcast(&_cXA(bi, bk, /*i +*/ x  , k, RMB));
                                                aa[x+1]   = _avx_broadcast(&_cXA(bi, bk, /*i +*/ x+1, k, RMB));
                                                dd0[x]   += bb0*aa[x];
                                                dd1[x]   += bb1*aa[x];
                                                dd0[x+1] += bb0*aa[x+1];
                                                dd1[x+1] += bb1*aa[x+1];
                                                }));
                                }), PREFETCH(&_bC(bi, bj, i, j),  _MM_HINT_T0 ));

                                UNROLL(x, RMB,
                                     ({
                                      *(__d4v *)(&_bC(bi, bj, i + x, j  )) += dd0[x] * alpha4;
                                      *(__d4v *)(&_bC(bi, bj, i + x, j+4)) += dd1[x] * alpha4;
                                     }));
                            }
                        }else{
                            packA_<RMB>(&_bA(bi, bk, i, 0), K, &pblkA[0], _RMB, _RMB, KB); // _c4AAAA(bi,bk,i,k)
                            for (int j = 0; j < _NB; j += 8) {
                                __d4v dd0[RMB], dd1[RMB];
                                PREFETCH(&_c8B(bk, bj, 0, j, 0),  _MM_HINT_T0 );
                                FOR(x, _RMB, ({dd0[x] = _avx0();dd1[x] = _avx0();}));
                                FORXX(k, _KB,({
                                    __d4v aa[RMB];
                                    __d4v bb0 = _avx_load_d4(&_c8B(bk, bj, k, j, 0));
                                    __d4v bb1 = _avx_load_d4(&_c8B(bk, bj, k, j, 4));
                                    FOR(x, _RMB, ({ 
                                                aa[x]   = _avx_broadcast(&_cXA(bi, bk, /*i +*/ x, k, RMB));
                                                dd0[x] += bb0*aa[x];
                                                dd1[x] += bb1*aa[x];}));
                                }), _mm_prefetch(&_bC(bi, bj, i, j),  _MM_HINT_T0 ));

                                FOR(x, _RMB, 
                                    ({
                                     *(__d4v *)(&_bC(bi, bj, i + x, j  )) += dd0[x] * alpha4;
                                     *(__d4v *)(&_bC(bi, bj, i + x, j+4)) += dd1[x] * alpha4;
                                    }));
                            }
                        }
                    }
                    // clang-format on
                }
            }
        }
    }
}

}  // namespace MM
