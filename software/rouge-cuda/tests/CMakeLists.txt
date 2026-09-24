add_executable(test_driver_vadd test_driver_vadd.cpp)
target_link_libraries(test_driver_vadd PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME driver_vadd COMMAND test_driver_vadd)

# Canonical kernels shared with the rouge-compiler AOT pipeline: one .ptx file,
# two execution paths (here — the interpreter; there — ptx2ir + native code).
# The headers are generated from the .ptx files so both paths execute
# byte-identical PTX.
set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-compiler/tests/kernels)
foreach(kern block_reduce atomic_reduce fp16_reduce gemm_tile)
  string(TOUPPER ${kern} KERN_UC)
  string(REPLACE "BLOCK_REDUCE" "BlockReduce" KERN_CAMEL ${KERN_UC})
  string(REPLACE "ATOMIC_REDUCE" "AtomicReduce" KERN_CAMEL ${KERN_CAMEL})
  string(REPLACE "FP16_REDUCE" "Fp16Reduce" KERN_CAMEL ${KERN_CAMEL})
  string(REPLACE "GEMM_TILE" "GemmTile" KERN_CAMEL ${KERN_CAMEL})
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

add_executable(test_interposer test_interposer.cpp)
target_compile_definitions(test_interposer PRIVATE ROUGE_SHARED_LIB="$<TARGET_FILE:rouge-cuda>")
target_link_libraries(test_interposer PRIVATE rouge-cuda-static ${CMAKE_DL_LIBS})
add_test(NAME interposer COMMAND test_interposer)
