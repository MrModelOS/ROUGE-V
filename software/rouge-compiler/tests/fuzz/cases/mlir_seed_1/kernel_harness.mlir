module {
  func.func @fuzz_kernel(%arg0: memref<256xi32>, %arg1: memref<64xi32>, %arg2: i32, %arg3: i32) {
    %alloc = memref.alloc() : memref<64xi32, 3 : i32>
    %0 = rouge.thread_id x : index
    %1 = arith.index_cast %0 : index to i32
    %2 = rouge.thread_id x : index
    %3 = memref.load %arg1[%2] : memref<64xi32>
    %4 = arith.addi %3, %arg3 : i32
    %5 = rouge.thread_id x : index
    memref.store %4, %alloc[%5] : memref<64xi32, 3 : i32>
    rouge.barrier
    %6 = arith.subi %arg2, %1 : i32
    %c1_i32 = arith.constant 1 : i32
    %7 = arith.subi %6, %c1_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = memref.load %alloc[%8] : memref<64xi32, 3 : i32>
    %c2_i32 = arith.constant 2 : i32
    %c93_i32 = arith.constant 93 : i32
    %10 = arith.muli %9, %c2_i32 : i32
    %11 = arith.addi %10, %c93_i32 : i32
    %12 = arith.xori %11, %arg3 : i32
    %c3_i32 = arith.constant 3 : i32
    %13 = arith.muli %11, %c3_i32 : i32
    %14 = arith.subi %12, %13 : i32
    %c4_i32 = arith.constant 4 : i32
    %15 = arith.muli %1, %c4_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %16 = arith.addi %15, %c0_i32 : i32
    %17 = arith.index_cast %16 : i32 to index
    memref.store %11, %arg0[%17] : memref<256xi32>
    %c4_i32_0 = arith.constant 4 : i32
    %18 = arith.muli %1, %c4_i32_0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %19 = arith.addi %18, %c1_i32_1 : i32
    %20 = arith.index_cast %19 : i32 to index
    memref.store %12, %arg0[%20] : memref<256xi32>
    %c4_i32_2 = arith.constant 4 : i32
    %21 = arith.muli %1, %c4_i32_2 : i32
    %c2_i32_3 = arith.constant 2 : i32
    %22 = arith.addi %21, %c2_i32_3 : i32
    %23 = arith.index_cast %22 : i32 to index
    memref.store %13, %arg0[%23] : memref<256xi32>
    %c4_i32_4 = arith.constant 4 : i32
    %24 = arith.muli %1, %c4_i32_4 : i32
    %c3_i32_5 = arith.constant 3 : i32
    %25 = arith.addi %24, %c3_i32_5 : i32
    %26 = arith.index_cast %25 : i32 to index
    memref.store %14, %arg0[%26] : memref<256xi32>
    return
  }
  func.func @fuzz_kernel_host(%arg0: memref<256xi32>, %arg1: memref<64xi32>, %arg2: i32, %arg3: i32) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    "rouge.launch"(%c1, %c1_0, %c1_0, %c64, %c64, %c64, %arg0, %arg1, %arg2, %arg3) <{kernel = @fuzz_kernel}> : (index, index, index, index, index, index, memref<256xi32>, memref<64xi32>, i32, i32) -> ()
    return
  }
  func.func private @print_i32(i32)
  func.func @main() {
    %out = memref.alloc() : memref<256xi32>
    %in = memref.alloc() : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %n = arith.constant 64 : i32
    %seed = arith.constant 12345 : i32
    %v0 = arith.constant 0 : i32
    %k0 = arith.constant 0 : index
    memref.store %v0, %in[%k0] : memref<64xi32>
    %v1 = arith.constant 1480645982 : i32
    %k1 = arith.constant 1 : index
    memref.store %v1, %in[%k1] : memref<64xi32>
    %v2 = arith.constant 955668050 : i32
    %k2 = arith.constant 2 : index
    memref.store %v2, %in[%k2] : memref<64xi32>
    %v3 = arith.constant 84853737 : i32
    %k3 = arith.constant 3 : index
    memref.store %v3, %in[%k3] : memref<64xi32>
    %v4 = arith.constant -1448634495 : i32
    %k4 = arith.constant 4 : index
    memref.store %v4, %in[%k4] : memref<64xi32>
    %v5 = arith.constant -437094558 : i32
    %k5 = arith.constant 5 : index
    memref.store %v5, %in[%k5] : memref<64xi32>
    %v6 = arith.constant 451858896 : i32
    %k6 = arith.constant 6 : index
    memref.store %v6, %in[%k6] : memref<64xi32>
    %v7 = arith.constant -2144010339 : i32
    %k7 = arith.constant 7 : index
    memref.store %v7, %in[%k7] : memref<64xi32>
    %v8 = arith.constant 2147483647 : i32
    %k8 = arith.constant 8 : index
    memref.store %v8, %in[%k8] : memref<64xi32>
    %v9 = arith.constant 1105483547 : i32
    %k9 = arith.constant 9 : index
    memref.store %v9, %in[%k9] : memref<64xi32>
    %v10 = arith.constant -1968188009 : i32
    %k10 = arith.constant 10 : index
    memref.store %v10, %in[%k10] : memref<64xi32>
    %v11 = arith.constant -1126590959 : i32
    %k11 = arith.constant 11 : index
    memref.store %v11, %in[%k11] : memref<64xi32>
    %v12 = arith.constant -770361762 : i32
    %k12 = arith.constant 12 : index
    memref.store %v12, %in[%k12] : memref<64xi32>
    %v13 = arith.constant 1755676021 : i32
    %k13 = arith.constant 13 : index
    memref.store %v13, %in[%k13] : memref<64xi32>
    %v14 = arith.constant -1350363458 : i32
    %k14 = arith.constant 14 : index
    memref.store %v14, %in[%k14] : memref<64xi32>
    %v15 = arith.constant -2118736626 : i32
    %k15 = arith.constant 15 : index
    memref.store %v15, %in[%k15] : memref<64xi32>
    %v16 = arith.constant 0 : i32
    %k16 = arith.constant 16 : index
    memref.store %v16, %in[%k16] : memref<64xi32>
    %v17 = arith.constant 1094887465 : i32
    %k17 = arith.constant 17 : index
    memref.store %v17, %in[%k17] : memref<64xi32>
    %v18 = arith.constant 1181104089 : i32
    %k18 = arith.constant 18 : index
    memref.store %v18, %in[%k18] : memref<64xi32>
    %v19 = arith.constant -1163024424 : i32
    %k19 = arith.constant 19 : index
    memref.store %v19, %in[%k19] : memref<64xi32>
    %v20 = arith.constant 1865537624 : i32
    %k20 = arith.constant 20 : index
    memref.store %v20, %in[%k20] : memref<64xi32>
    %v21 = arith.constant 867440552 : i32
    %k21 = arith.constant 21 : index
    memref.store %v21, %in[%k21] : memref<64xi32>
    %v22 = arith.constant 384102704 : i32
    %k22 = arith.constant 22 : index
    memref.store %v22, %in[%k22] : memref<64xi32>
    %v23 = arith.constant -498595516 : i32
    %k23 = arith.constant 23 : index
    memref.store %v23, %in[%k23] : memref<64xi32>
    %v24 = arith.constant 0 : i32
    %k24 = arith.constant 24 : index
    memref.store %v24, %in[%k24] : memref<64xi32>
    %v25 = arith.constant 840927711 : i32
    %k25 = arith.constant 25 : index
    memref.store %v25, %in[%k25] : memref<64xi32>
    %v26 = arith.constant 7721081 : i32
    %k26 = arith.constant 26 : index
    memref.store %v26, %in[%k26] : memref<64xi32>
    %v27 = arith.constant 1220804323 : i32
    %k27 = arith.constant 27 : index
    memref.store %v27, %in[%k27] : memref<64xi32>
    %v28 = arith.constant -1211940493 : i32
    %k28 = arith.constant 28 : index
    memref.store %v28, %in[%k28] : memref<64xi32>
    %v29 = arith.constant -272476916 : i32
    %k29 = arith.constant 29 : index
    memref.store %v29, %in[%k29] : memref<64xi32>
    %v30 = arith.constant -1973268975 : i32
    %k30 = arith.constant 30 : index
    memref.store %v30, %in[%k30] : memref<64xi32>
    %v31 = arith.constant -589346422 : i32
    %k31 = arith.constant 31 : index
    memref.store %v31, %in[%k31] : memref<64xi32>
    %v32 = arith.constant 2147483647 : i32
    %k32 = arith.constant 32 : index
    memref.store %v32, %in[%k32] : memref<64xi32>
    %v33 = arith.constant -786504876 : i32
    %k33 = arith.constant 33 : index
    memref.store %v33, %in[%k33] : memref<64xi32>
    %v34 = arith.constant 1836387913 : i32
    %k34 = arith.constant 34 : index
    memref.store %v34, %in[%k34] : memref<64xi32>
    %v35 = arith.constant 1798509290 : i32
    %k35 = arith.constant 35 : index
    memref.store %v35, %in[%k35] : memref<64xi32>
    %v36 = arith.constant -1481734595 : i32
    %k36 = arith.constant 36 : index
    memref.store %v36, %in[%k36] : memref<64xi32>
    %v37 = arith.constant -283214773 : i32
    %k37 = arith.constant 37 : index
    memref.store %v37, %in[%k37] : memref<64xi32>
    %v38 = arith.constant -1131633585 : i32
    %k38 = arith.constant 38 : index
    memref.store %v38, %in[%k38] : memref<64xi32>
    %v39 = arith.constant 2029512098 : i32
    %k39 = arith.constant 39 : index
    memref.store %v39, %in[%k39] : memref<64xi32>
    %v40 = arith.constant -2147483648 : i32
    %k40 = arith.constant 40 : index
    memref.store %v40, %in[%k40] : memref<64xi32>
    %v41 = arith.constant -19611433 : i32
    %k41 = arith.constant 41 : index
    memref.store %v41, %in[%k41] : memref<64xi32>
    %v42 = arith.constant -559151749 : i32
    %k42 = arith.constant 42 : index
    memref.store %v42, %in[%k42] : memref<64xi32>
    %v43 = arith.constant -1013074672 : i32
    %k43 = arith.constant 43 : index
    memref.store %v43, %in[%k43] : memref<64xi32>
    %v44 = arith.constant -1810977905 : i32
    %k44 = arith.constant 44 : index
    memref.store %v44, %in[%k44] : memref<64xi32>
    %v45 = arith.constant -495383377 : i32
    %k45 = arith.constant 45 : index
    memref.store %v45, %in[%k45] : memref<64xi32>
    %v46 = arith.constant 170000569 : i32
    %k46 = arith.constant 46 : index
    memref.store %v46, %in[%k46] : memref<64xi32>
    %v47 = arith.constant 1040951910 : i32
    %k47 = arith.constant 47 : index
    memref.store %v47, %in[%k47] : memref<64xi32>
    %v48 = arith.constant 0 : i32
    %k48 = arith.constant 48 : index
    memref.store %v48, %in[%k48] : memref<64xi32>
    %v49 = arith.constant -1379957770 : i32
    %k49 = arith.constant 49 : index
    memref.store %v49, %in[%k49] : memref<64xi32>
    %v50 = arith.constant -172598850 : i32
    %k50 = arith.constant 50 : index
    memref.store %v50, %in[%k50] : memref<64xi32>
    %v51 = arith.constant 1996438266 : i32
    %k51 = arith.constant 51 : index
    memref.store %v51, %in[%k51] : memref<64xi32>
    %v52 = arith.constant -885604635 : i32
    %k52 = arith.constant 52 : index
    memref.store %v52, %in[%k52] : memref<64xi32>
    %v53 = arith.constant 644528993 : i32
    %k53 = arith.constant 53 : index
    memref.store %v53, %in[%k53] : memref<64xi32>
    %v54 = arith.constant -1133062747 : i32
    %k54 = arith.constant 54 : index
    memref.store %v54, %in[%k54] : memref<64xi32>
    %v55 = arith.constant 2090271985 : i32
    %k55 = arith.constant 55 : index
    memref.store %v55, %in[%k55] : memref<64xi32>
    %v56 = arith.constant -2147483648 : i32
    %k56 = arith.constant 56 : index
    memref.store %v56, %in[%k56] : memref<64xi32>
    %v57 = arith.constant -1689752474 : i32
    %k57 = arith.constant 57 : index
    memref.store %v57, %in[%k57] : memref<64xi32>
    %v58 = arith.constant -1407028615 : i32
    %k58 = arith.constant 58 : index
    memref.store %v58, %in[%k58] : memref<64xi32>
    %v59 = arith.constant -786270723 : i32
    %k59 = arith.constant 59 : index
    memref.store %v59, %in[%k59] : memref<64xi32>
    %v60 = arith.constant -2070727766 : i32
    %k60 = arith.constant 60 : index
    memref.store %v60, %in[%k60] : memref<64xi32>
    %v61 = arith.constant 665251955 : i32
    %k61 = arith.constant 61 : index
    memref.store %v61, %in[%k61] : memref<64xi32>
    %v62 = arith.constant 1324617531 : i32
    %k62 = arith.constant 62 : index
    memref.store %v62, %in[%k62] : memref<64xi32>
    %v63 = arith.constant -2108585201 : i32
    %k63 = arith.constant 63 : index
    memref.store %v63, %in[%k63] : memref<64xi32>
    func.call @fuzz_kernel_host(%out, %in, %n, %seed) : (memref<256xi32>, memref<64xi32>, i32, i32) -> ()
    scf.for %i = %c0 to %c256 step %c1 {
      %v = memref.load %out[%i] : memref<256xi32>
      func.call @print_i32(%v) : (i32) -> ()
    }
    memref.dealloc %out : memref<256xi32>
    memref.dealloc %in : memref<64xi32>
    return
  }
}
