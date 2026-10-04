// Сверка раскладки rouge_cuda_func_attributes с настоящим cudaFuncAttributes.
//
// Тот же принцип, что в test_device_prop_layout.cpp: наш заголовок —
// ABI-контракт, offsets обязаны совпадать. Настоящая структура извлекается
// из driver_types.h на этапе configure (см. gen_real_prop_header.py).
// Без настоящих заголовков тест не собирается (SKIP на уровне CMake).

#include <stddef.h>
#include <stdio.h>

#include <rouge/cuda_runtime_api.h>

#include "real_cuda_func_attributes.h"

#define CHECK_FIELD(field)                                                                \
  do {                                                                                    \
    if (offsetof(rouge_cuda_func_attributes, field) != offsetof(cudaFuncAttributes, field) || \
        sizeof(((rouge_cuda_func_attributes*)0)->field) !=                                   \
            sizeof(((cudaFuncAttributes*)0)->field)) {                                     \
      printf("FAIL: field " #field ": ours off=%zu size=%zu, real off=%zu size=%zu\n",    \
             offsetof(rouge_cuda_func_attributes, field),                                    \
             sizeof(((rouge_cuda_func_attributes*)0)->field),                                \
             offsetof(cudaFuncAttributes, field), sizeof(((cudaFuncAttributes*)0)->field)); \
      ++bad;                                                                               \
    }                                                                                      \
  } while (0)

int main() {
  int bad = 0;
  if (sizeof(rouge_cuda_func_attributes) != sizeof(cudaFuncAttributes)) {
    printf("FAIL: sizeof ours=%zu real=%zu\n", sizeof(rouge_cuda_func_attributes),
           sizeof(cudaFuncAttributes));
    ++bad;
  }

  CHECK_FIELD(sharedSizeBytes);
  CHECK_FIELD(constSizeBytes);
  CHECK_FIELD(localSizeBytes);
  CHECK_FIELD(maxThreadsPerBlock);
  CHECK_FIELD(numRegs);
  CHECK_FIELD(ptxVersion);
  CHECK_FIELD(binaryVersion);
  CHECK_FIELD(cacheModeCA);
  CHECK_FIELD(maxDynamicSharedSizeBytes);
  CHECK_FIELD(preferredShmemCarveout);
  CHECK_FIELD(clusterDimMustBeSet);
  CHECK_FIELD(requiredClusterWidth);
  CHECK_FIELD(requiredClusterHeight);
  CHECK_FIELD(requiredClusterDepth);
  CHECK_FIELD(clusterSchedulingPolicyPreference);
  CHECK_FIELD(nonPortableClusterSizeAllowed);
  CHECK_FIELD(deviceNodeUpdateStatus);
  CHECK_FIELD(sharedMemoryMode);
  CHECK_FIELD(reserved);

  if (bad == 0)
    printf("PASS: rouge_cuda_func_attributes matches the real cudaFuncAttributes layout\n");
  return bad ? 1 : 0;
}
