# ROUGE-V differential fuzzer.
#
# Три независимых пути исполнения одного и того же ядра:
#   * наш текстовый интерпретатор PTX (эталон семантики);
#   * наш AOT-нижнинг -> LLVM IR -> исполнение на хосте;
#   * наш AOT-нижнинг -> LLVM IR -> PTX для GPU -> настоящая карта.
#
# Карта — арбитр: где наше представление о семантике PTX расходится с
# железом, права у железа. Именно так нашёлся баг с полем селектора в
# prmt (сдвиг влево вместо вправо) и разошлись знаковости dp4a.
#
# Расхождение сохраняется в cases/ — оно становится постоянным регрессионным
# тестом, а не флаком.

find_package(Python3 COMPONENTS Interpreter QUIET)

# ---- генератор случайных ядер -------------------------------------------
add_executable(fuzz_gen_kernel gen_kernel.cpp)

# ---- эталон (интерпретатор) и хостовая половина ---------------------------
add_executable(fuzz_run_interp run_interp.cpp)
target_link_libraries(fuzz_run_interp PRIVATE rouge-ptx)
target_include_directories(fuzz_run_interp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)

# Хостовая половина (run_host.cpp) намеренно НЕ собирается здесь: раннер
# линкуется с AOT-объектом ядра на каждый seed, поэтому его собирает
# драйвер цикла (run_fuzz.py --host-runner ...).

# ---- арбитр (карта) -------------------------------------------------------
if(CUDA_DRIVER_INCLUDE_DIR AND CUDA_DRIVER_LIBRARY)
  add_executable(fuzz_run_gpu run_gpu.c)
  target_include_directories(fuzz_run_gpu PRIVATE ${CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(fuzz_run_gpu PRIVATE ${CUDA_DRIVER_LIBRARY})
endif()

# ---- драйвер цикла -------------------------------------------------------
if(Python3_Interpreter_FOUND AND CLANG_EXECUTABLE)
  add_test(NAME compiler_ptx_fuzz
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_fuzz.py
            --ptx2ir $<TARGET_FILE:ptx2ir>
            --clang ${CLANG_EXECUTABLE}
            --gen $<TARGET_FILE:fuzz_gen_kernel>
            --interp $<TARGET_FILE:fuzz_run_interp>
            --host-runner ${CMAKE_CURRENT_SOURCE_DIR}/run_host.cpp
            --mode host
            --iters 8
            --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work
            --cases ${CMAKE_CURRENT_SOURCE_DIR}/cases)
  set_tests_properties(compiler_ptx_fuzz PROPERTIES TIMEOUT 900)

  if(TARGET fuzz_run_gpu)
    add_test(NAME compiler_ptx_fuzz_gpu
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_fuzz.py
              --ptx2ir $<TARGET_FILE:ptx2ir>
              --clang ${CLANG_EXECUTABLE}
              --gen $<TARGET_FILE:fuzz_gen_kernel>
              --interp $<TARGET_FILE:fuzz_run_interp>
              --gpu $<TARGET_FILE:fuzz_run_gpu>
              --mode gpu
              --iters 16
              --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work-gpu
              --cases ${CMAKE_CURRENT_SOURCE_DIR}/cases)
    set_tests_properties(compiler_ptx_fuzz_gpu PROPERTIES TIMEOUT 900)
  endif()
endif()
