// Occupancy для референсного sm_75: сколько блоков ядра одновременно живёт
// на одном SM.
//
// Алгоритм — документированный, из открытого cuda_occupancy.h (поставляется с
// тулкитом): ответ есть минимум лимитов (варпы, регистры, блоки, shared).
// Partitioned global caching на sm_75 не поддерживается (только 5.2/5.3/6.x),
// поэтому ветки GC здесь нет — это следует из того же заголовка, а не из
// предположения.
//
// Все архитектурные константы ниже — для sm_75 (Turing, референсный таргет),
// их значения сверены с настоящим драйвером MX450 пробой (см. probe_props и
// probe_occ в истории коммитов), а не взяты по памяти:
//   maxThreadsPerSM 1024, maxBlocksPerSM 16, regsPerSM 65536,
//   regAllocGranularity 256, maxRegsPerThread 256 (таблица заголовка),
//   smemAllocGranularity 256, default carveout 65536, per-block limit 49152.

#pragma once

#include "ptx.h"

#include <cstddef>

namespace rouge {

// Статистика ядра, из которой считается occupancy. numRegs — ОЦЕНКА числа
// живых 32-битных регистров на поток (max-live анализ тела, см. ниже):
// настоящий аллокатор ptxas нам недоступен (его данные лежат в
// недокументированном .nv.info — это вне legal frame), поэтому точное число
// взять негде. Max-live обычно попадает в тот же бакет аллокации 256, и
// тогда occupancy совпадает с драйвером побитово (проверено сравнением
// wave1_occupancy на настоящем стеке). Если тело содержит call или анализ
// не сошёлся — откат к сумме .reg-деклараций (безопасная верхняя граница:
// завышение регистров может только занизить ответ).
struct KernelStats {
  int numRegs = 0;          // верхняя граница 32-битных регистров на поток
  size_t staticSmem = 0;     // .shared байт на блок (fn.sharedSize)
  size_t localSize = 0;      // .local байт на поток (fn.localSize)
  size_t constSize = 0;      // .const байт модуля
  int targetSm = 0;          // .target sm_XX (0, если в тексте его нет)
};

// Собрать статистику из распарсенного модуля. Возвращает false, если индекс
// функции вне диапазона.
bool kernel_stats(const PtxProgram& prog, int fnIndex, KernelStats* out);

// Находит (программу, индекс) по mangled-имени ядра: сначала runtime-модули
// (ROUGE_CUDA_PTX + fatbin), потом driver-модули (cuModuleLoadData).
bool find_kernel_program(const std::string& name, const PtxProgram** prog, int* fnIndex);

// Активных блоков на SM (0 = такая конфигурация не запускается; это не
// ошибка — настоящий драйвер тоже возвращает успех с нулём).
int occ_active_blocks(const KernelStats& st, int blockSize, size_t dynamicSmem);

// Предложенная конфигурация: перебор размеров блока от 1024 вниз с шагом 32,
// максимум потоков, при равенстве — больший блок (документированный алгоритм
// configurator'а). numSms — число SM машины (у нас 1, см. device props).
void occ_potential_block_size(int* minGridSize, int* blockSize, const KernelStats& st,
                              size_t (*blockSizeToDynamicSMem)(int), size_t dynamicSmemFixed,
                              int numSms);

// Доступный dynamic shared на блок при numBlocks блоках на SM.
// Возвращает false (и оставляет *out нетронутым, как настоящий драйвер),
// если конфигурация невозможна.
bool occ_available_dynamic_smem(size_t* out, const KernelStats& st, int numBlocks,
                                int blockSize);

}  // namespace rouge
