// Шимы cuBLAS/cuDNN: счёт через наш код против независимых эталонов.
//
// cublasSgemm_v2: все 4 комбинации N/T на случайных матрицах против
//   эталона в double (column-major, с alpha/beta). Плюс Create/Destroy,
//   версия, статусы, отказ на мусорных lda.
// cudnnConvolutionForward: NCHW f32 против rouge_conv2d_f32, оба алгоритма
//   (IMPLICIT_GEMM и WINOGRAD), alpha/beta-масштабирование, dims,
//   отказ на HALF и на dilation.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/cublas_shim.h"
#include "rouge/cudnn_shim.h"
#include "rouge/numerics.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

// Эталон SGEMM в double, column-major.
static void ref_sgemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                      int k, float al, const std::vector<float> &A, int lda,
                      const std::vector<float> &B, int ldb, float be,
                      std::vector<float> &C, int ldc) {
  auto opA = [&](int i, int j) -> double {
    return (ta == CUBLAS_OP_N) ? A[(size_t)j * lda + i] : A[(size_t)i * lda + j];
  };
  auto opB = [&](int i, int j) -> double {
    return (tb == CUBLAS_OP_N) ? B[(size_t)j * ldb + i] : B[(size_t)i * ldb + j];
  };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p) s += opA(i, p) * opB(p, j);
      C[(size_t)j * ldc + i] = (float)(al * s + be * C[(size_t)j * ldc + i]);
    }
}

int main() {
  std::mt19937 rng(0x5A47u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // --- cuBLAS ---
  cublasHandle_t h = nullptr;
  EXPECT_TRUE(cublasCreate_v2(&h) == CUBLAS_STATUS_SUCCESS && h);
  EXPECT_TRUE(cublasCreate_v2(nullptr) == CUBLAS_STATUS_INVALID_VALUE);
  int ver = 0;
  EXPECT_TRUE(cublasGetVersion_v2(h, &ver) == CUBLAS_STATUS_SUCCESS);
  EXPECT_TRUE(ver == 200);
  EXPECT_TRUE(cublasGetStatusString(CUBLAS_STATUS_SUCCESS) != nullptr);

  const int m = 5, n = 4, k = 6;
  for (int combo = 0; combo < 4; ++combo) {
    const auto ta =
        (combo & 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto tb =
        (combo & 2) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = (ta == CUBLAS_OP_N ? m : k) + 1;  // с запасом
    const int ldb = (tb == CUBLAS_OP_N ? k : n) + 1;
    const int ldc = m + 1;
    std::vector<float> A((size_t)lda * (ta == CUBLAS_OP_N ? k : m));
    std::vector<float> B((size_t)ldb * (tb == CUBLAS_OP_N ? n : k));
    std::vector<float> C((size_t)ldc * n), ref = C;
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (auto &v : C) v = dist(rng);
    ref = C;
    const float al = 0.5f, be = 1.5f;
    EXPECT_TRUE(cublasSgemm_v2(h, ta, tb, m, n, k, &al, A.data(), lda,
                               B.data(), ldb, &be, C.data(), ldc) ==
                CUBLAS_STATUS_SUCCESS);
    ref_sgemm(ta, tb, m, n, k, al, A, lda, B, ldb, be, ref, ldc);
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      // Сравниваем только рабочую m×n область (хвосты lda/ldc — мусор).
      md = std::max(md, std::fabs((double)C[i] - ref[i]));
      sc = std::max(sc, std::fabs((double)ref[i]));
    }
    if (md / sc > 1e-5) {
      std::printf("sgemm combo %d rel %g\n", combo, md / sc);
      ++failures;
    }
  }
  {
    float a = 1, b = 0;
    std::vector<float> A(4, 1), B(4, 1), C(4, 0);
    EXPECT_TRUE(cublasSgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &a,
                               A.data(), 1, B.data(), 2, &b, C.data(), 2) ==
                CUBLAS_STATUS_INVALID_VALUE);  // lda < m
    EXPECT_TRUE(cublasSgemm_v2(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &a,
                               A.data(), 2, B.data(), 2, &b, C.data(), 2) ==
                CUBLAS_STATUS_NOT_INITIALIZED);
  }
  EXPECT_TRUE(cublasDestroy_v2(h) == CUBLAS_STATUS_SUCCESS);

  // --- cuDNN ---
  cudnnHandle_t dh = nullptr;
  EXPECT_TRUE(cudnnCreate(&dh) == CUDNN_STATUS_SUCCESS && dh);
  cudnnTensorDescriptor_t xD = nullptr, yD = nullptr;
  cudnnFilterDescriptor_t wD = nullptr;
  cudnnConvolutionDescriptor_t conv = nullptr;
  cudnnCreateTensorDescriptor(&xD);
  cudnnCreateTensorDescriptor(&yD);
  cudnnCreateFilterDescriptor(&wD);
  cudnnCreateConvolutionDescriptor(&conv);
  EXPECT_TRUE(cudnnSetTensor4dDescriptor(xD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         1, 2, 8, 8) == CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(cudnnSetFilter4dDescriptor(wD, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 3, 2, 3, 3) ==
              CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(cudnnSetConvolution2dDescriptor(conv, 1, 1, 1, 1, 1, 1, 0,
                                              CUDNN_DATA_FLOAT) ==
              CUDNN_STATUS_SUCCESS);
  int on, oc, oh, ow;
  EXPECT_TRUE(cudnnGetConvolution2dForwardOutputDim(conv, xD, wD, &on, &oc, &oh,
                                                    &ow) == CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(on == 1 && oc == 3 && oh == 8 && ow == 8);
  EXPECT_TRUE(cudnnSetTensor4dDescriptor(yD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         on, oc, oh, ow) == CUDNN_STATUS_SUCCESS);
  std::vector<float> X(1 * 2 * 8 * 8), W(3 * 2 * 9), Y(1 * 3 * 8 * 8, 0.25f),
      ref(Y.size(), 0);
  for (auto &v : X) v = dist(rng);
  for (auto &v : W) v = dist(rng);
  ref = Y;
  const float al = 2.0f, be = 0.5f;
  for (int algo = 0; algo < 2; ++algo) {
    std::vector<float> got = Y;
    const auto alg = algo == 0 ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
                               : CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    EXPECT_TRUE(cudnnConvolutionForward(dh, &al, xD, X.data(), wD, W.data(),
                                        conv, alg, nullptr, 0, &be, yD,
                                        got.data()) == CUDNN_STATUS_SUCCESS);
    std::vector<float> base(ref.size(), 0);
    rouge_conv2d_f32(1, 2, 8, 8, 3, 3, 3, 1, 1, X.data(), W.data(), nullptr,
                     base.data(), nullptr);
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double want = al * base[i] + be * ref[i];
      md = std::max(md, std::fabs((double)got[i] - want));
      sc = std::max(sc, std::fabs(want));
    }
    if (md / sc > 1e-4) {
      std::printf("cudnn algo %d rel %g\n", algo, md / sc);
      ++failures;
    }
  }
  // Отказы: HALF-тензор и dilation 2.
  cudnnTensorDescriptor_t halfD = nullptr;
  cudnnCreateTensorDescriptor(&halfD);
  cudnnSetTensor4dDescriptor(halfD, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 2, 8,
                             8);
  EXPECT_TRUE(cudnnConvolutionForward(dh, &al, halfD, X.data(), wD, W.data(),
                                      conv,
                                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                      nullptr, 0, &be, yD, Y.data()) ==
              CUDNN_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(cudnnSetConvolution2dDescriptor(conv, 1, 1, 1, 1, 2, 2, 0,
                                              CUDNN_DATA_FLOAT) ==
              CUDNN_STATUS_NOT_SUPPORTED);
  cudnnDestroyTensorDescriptor(xD);
  cudnnDestroyTensorDescriptor(yD);
  cudnnDestroyTensorDescriptor(halfD);
  cudnnDestroyFilterDescriptor(wD);
  cudnnDestroyConvolutionDescriptor(conv);
  EXPECT_TRUE(cudnnDestroy(dh) == CUDNN_STATUS_SUCCESS);

  if (failures == 0) std::printf("PASS: cublas + cudnn shims\n");
  return failures ? 1 : 0;
}
