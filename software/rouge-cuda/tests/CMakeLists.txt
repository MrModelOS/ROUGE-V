add_executable(test_driver_vadd test_driver_vadd.cpp)
target_link_libraries(test_driver_vadd PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME driver_vadd COMMAND test_driver_vadd)

# Canonical kernels shared with the rouge-compiler AOT pipeline: one .ptx file,
# two execution paths (here — the interpreter; there — ptx2ir + native code).
# The headers are generated from the .ptx files so both paths execute
# byte-identical PTX.
set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-compiler/tests/kernels)
foreach(kern block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas)
  string(TOUPPER ${kern} KERN_UC)
  string(REPLACE "BLOCK_REDUCE" "BlockReduce" KERN_CAMEL ${KERN_UC})
  string(REPLACE "ATOMIC_REDUCE" "AtomicReduce" KERN_CAMEL ${KERN_CAMEL})
  string(REPLACE "FP16_REDUCE" "Fp16Reduce" KERN_CAMEL ${KERN_CAMEL})
  string(REPLACE "GEMM_TILE" "GemmTile" KERN_CAMEL ${KERN_CAMEL})
  string(REPLACE "ATOM_CAS" "AtomCas" KERN_CAMEL ${KERN_CAMEL})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kern}_ptx.h
    COMMAND ${CMAKE_COMMAND}
      -DPTX_FILE=${KERNEL_DIR}/${kern}.ptx
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${kern}_ptx.h
      -DVAR_NAME=k${KERN_CAMEL}Ptx
      -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_ptx.cmake
    DEPENDS ${KERNEL_DIR}/${kern}.ptx ${CMAKE_CURRENT_SOURCE_DIR}/embed_ptx.cmake)
  add_custom_target(${kern}_ptx_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${kern}_ptx.h)
  add_executable(test_driver_${kern} test_driver_${kern}.cpp)
  target_include_directories(test_driver_${kern} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(test_driver_${kern} ${kern}_ptx_header)
  target_link_libraries(test_driver_${kern} PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME driver_${kern} COMMAND test_driver_${kern})
endforeach()

add_executable(test_runtime_api test_runtime_api.cpp)
target_link_libraries(test_runtime_api PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME runtime_api COMMAND test_runtime_api)

add_executable(test_driver_device_global test_driver_device_global.cpp)
target_link_libraries(test_driver_device_global PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME driver_device_global COMMAND test_driver_device_global)

add_executable(test_interposer test_interposer.cpp)
target_compile_definitions(test_interposer PRIVATE ROUGE_SHARED_LIB="$<TARGET_FILE:rouge-cuda>")
target_link_libraries(test_interposer PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME interposer COMMAND test_interposer)

# ---- Раскладка cudaDeviceProp: сверка с настоящим заголовком NVIDIA -------
# Наш заголовок — ABI-контракт, поэтому offsets обязаны совпадать байт в
# байт. Настоящая структура извлекается из driver_types.h на этапе configure
# (только структура, без остального заголовка — он конфликтовал бы с нашим).
# Без настоящих заголовков или без python3 тест пропускается, а не падает.
find_path(ROUGE_REAL_CUDA_INCLUDE driver_types.h
  PATHS /opt/cuda/include /usr/local/cuda/include /usr/include)
find_program(ROUGE_PYTHON3 python3)
if(ROUGE_REAL_CUDA_INCLUDE AND ROUGE_PYTHON3)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_device_prop.h
    COMMAND ${ROUGE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/gen_real_prop_header.py
            ${ROUGE_REAL_CUDA_INCLUDE}/driver_types.h cudaDeviceProp
            ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_device_prop.h
    DEPENDS ${ROUGE_REAL_CUDA_INCLUDE}/driver_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_real_prop_header.py)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_func_attributes.h
    COMMAND ${ROUGE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/gen_real_prop_header.py
            ${ROUGE_REAL_CUDA_INCLUDE}/driver_types.h cudaFuncAttributes
            ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_func_attributes.h cudaFuncAttributes
    DEPENDS ${ROUGE_REAL_CUDA_INCLUDE}/driver_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_real_prop_header.py)
  add_custom_target(real_prop_header
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_device_prop.h
            ${CMAKE_CURRENT_BINARY_DIR}/real_cuda_func_attributes.h)
  add_executable(test_device_prop_layout test_device_prop_layout.cpp)
  target_include_directories(test_device_prop_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(test_device_prop_layout real_prop_header)
  add_test(NAME device_prop_layout COMMAND test_device_prop_layout)
  add_executable(test_func_attr_layout test_func_attr_layout.cpp)
  target_include_directories(test_func_attr_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(test_func_attr_layout real_prop_header)
  add_test(NAME func_attr_layout COMMAND test_func_attr_layout)

  # Юнит-тест occupancy-калькулятора (числа сверены с настоящим драйвером).
  # Внутренний заголовок occupancy.h виден только здесь, не устанавливается.
  add_executable(test_occupancy test_occupancy.cpp)
  target_include_directories(test_occupancy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_link_libraries(test_occupancy PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME occupancy COMMAND test_occupancy)

  # Driver-путь occupancy (CUfunction -> калькулятор).
  add_executable(test_driver_occupancy test_driver_occupancy.cpp)
  target_link_libraries(test_driver_occupancy PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME driver_occupancy COMMAND test_driver_occupancy)

  # mma.sync: каноническое ядро против хост-эталона (побитово).
  add_executable(test_mma test_mma.cpp)
  target_include_directories(test_mma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_link_libraries(test_mma PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME mma COMMAND test_mma ${KERNEL_DIR})

  # ldmatrix/stmatrix: roundtrip + transpose + x4.
  add_executable(test_ldmatrix test_ldmatrix.cpp)
  target_include_directories(test_ldmatrix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_link_libraries(test_ldmatrix PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME ldmatrix COMMAND test_ldmatrix ${KERNEL_DIR})

  # Narrow float форматы: исчерпывающая проверка через настоящий cvt.
  add_executable(test_narrow_fp test_narrow_fp.cpp)
  target_include_directories(test_narrow_fp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_link_libraries(test_narrow_fp PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
  add_test(NAME narrow_fp COMMAND test_narrow_fp)
else()
  message(STATUS "device_prop_layout: SKIP (needs real CUDA headers and python3)")
endif()

# ---- Волна 1: целая чужая программа, собранная настоящим nvcc --------------
# The program source lives in programs/ and is compiled by nvcc at test time.
# Without nvcc the test reports SKIP; it never needs a GPU.
# nvcc usually lives outside PATH (each toolkit installs it in its own bin),
# so search PATH first and then probe the standard locations explicitly.
find_program(ROUGE_NVCC_PROG nvcc)
if(NOT RROUGE_NVCC_PROG)
  foreach(_cuda_bin /opt/cuda/bin /usr/local/cuda/bin /usr/lib/nvidia-cuda-toolkit/bin)
    if(EXISTS "${_cuda_bin}/nvcc")
      set(ROUGE_NVCC_PROG "${_cuda_bin}/nvcc")
      break()
    endif()
  endforeach()
endif()
if(ROUGE_NVCC_PROG)
  add_executable(test_foreign_program test_foreign_program.cpp)
  target_link_libraries(test_foreign_program PRIVATE ${CMAKE_DL_LIBS})
  set(_fp_dir ${CMAKE_CURRENT_SOURCE_DIR}/programs)
  set(_fp_work ${CMAKE_CURRENT_BINARY_DIR}/foreign-program)
  add_test(NAME foreign_program
    COMMAND ${CMAKE_COMMAND}
      -DNVCC=${ROUGE_NVCC_PROG}
      "-DPROGRAMS=${_fp_dir}/wave1_vadd.cu;${_fp_dir}/wave1_api.cu;${_fp_dir}/wave1_occupancy.cu"
      -DWORK_DIR=${_fp_work}
      -DSHIM=$<TARGET_FILE:rouge-cuda>
      -DTEST_BIN=$<TARGET_FILE:test_foreign_program>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/foreign_program.cmake)
else()
  message(STATUS "foreign_program: SKIP (nvcc not found)")
endif()
