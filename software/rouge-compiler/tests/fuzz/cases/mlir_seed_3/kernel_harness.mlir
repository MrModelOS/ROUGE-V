module {
  func.func @fuzz_kernel(%arg0: memref<256xi32>, %arg1: memref<64xi32>, %arg2: i32, %arg3: i32) {
    %0 = rouge.block_id x : index
    %1 = arith.index_cast %0 : index to i32
    %2 = rouge.block_dim x : index
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.extui %1 : i32 to i64
    %5 = arith.extui %3 : i32 to i64
    %6 = arith.muli %4, %5 : i64
    %7 = rouge.thread_id x : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.extui %8 : i32 to i64
    %10 = arith.addi %6, %9 : i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.cmpi uge, %11, %arg2 : i32
    %true = arith.constant true
    %13 = arith.xori %12, %true : i1
    scf.if %13 {
      %14 = arith.index_cast %11 : i32 to index
      %15 = memref.load %arg1[%14] : memref<64xi32>
      %c0_i32 = arith.constant 0 : i32
      %16 = arith.extsi %c0_i32 : i32 to i64
      %c255_i32 = arith.constant 255 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %17 = arith.shrui %15, %c0_i32_0 : i32
      %18 = arith.andi %17, %c255_i32 : i32
      %c0_i32_1 = arith.constant 0 : i32
      %19 = arith.shrui %arg3, %c0_i32_1 : i32
      %20 = arith.andi %19, %c255_i32 : i32
      %21 = arith.extsi %20 : i32 to i64
      %22 = arith.extsi %18 : i32 to i64
      %23 = arith.muli %22, %21 : i64
      %24 = arith.addi %16, %23 : i64
      %c8_i32 = arith.constant 8 : i32
      %25 = arith.shrui %15, %c8_i32 : i32
      %26 = arith.andi %25, %c255_i32 : i32
      %c8_i32_2 = arith.constant 8 : i32
      %27 = arith.shrui %arg3, %c8_i32_2 : i32
      %28 = arith.andi %27, %c255_i32 : i32
      %29 = arith.extsi %28 : i32 to i64
      %30 = arith.extsi %26 : i32 to i64
      %31 = arith.muli %30, %29 : i64
      %32 = arith.addi %24, %31 : i64
      %c16_i32 = arith.constant 16 : i32
      %33 = arith.shrui %15, %c16_i32 : i32
      %34 = arith.andi %33, %c255_i32 : i32
      %c16_i32_3 = arith.constant 16 : i32
      %35 = arith.shrui %arg3, %c16_i32_3 : i32
      %36 = arith.andi %35, %c255_i32 : i32
      %37 = arith.extsi %36 : i32 to i64
      %38 = arith.extsi %34 : i32 to i64
      %39 = arith.muli %38, %37 : i64
      %40 = arith.addi %32, %39 : i64
      %c24_i32 = arith.constant 24 : i32
      %41 = arith.shrui %15, %c24_i32 : i32
      %42 = arith.andi %41, %c255_i32 : i32
      %c24_i32_4 = arith.constant 24 : i32
      %43 = arith.shrui %arg3, %c24_i32_4 : i32
      %44 = arith.andi %43, %c255_i32 : i32
      %45 = arith.extsi %44 : i32 to i64
      %46 = arith.extsi %42 : i32 to i64
      %47 = arith.muli %46, %45 : i64
      %48 = arith.addi %40, %47 : i64
      %49 = arith.trunci %48 : i64 to i32
      %50 = arith.extui %15 : i32 to i64
      %51 = arith.extui %arg3 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %52 = arith.shli %51, %c32_i64 : i64
      %53 = arith.ori %50, %52 : i64
      %c0_i32_5 = arith.constant 0 : i32
      %c40_i64 = arith.constant 40 : i64
      %54 = arith.shrui %53, %c40_i64 : i64
      %55 = arith.trunci %54 : i64 to i32
      %c0_i32_6 = arith.constant 0 : i32
      %56 = arith.shli %55, %c0_i32_6 : i32
      %57 = arith.ori %c0_i32_5, %56 : i32
      %c0_i64 = arith.constant 0 : i64
      %58 = arith.shrui %53, %c0_i64 : i64
      %59 = arith.trunci %58 : i64 to i32
      %c8_i32_7 = arith.constant 8 : i32
      %60 = arith.shli %59, %c8_i32_7 : i32
      %61 = arith.ori %57, %60 : i32
      %c0_i64_8 = arith.constant 0 : i64
      %62 = arith.shrui %53, %c0_i64_8 : i64
      %63 = arith.trunci %62 : i64 to i32
      %c16_i32_9 = arith.constant 16 : i32
      %64 = arith.shli %63, %c16_i32_9 : i32
      %65 = arith.ori %61, %64 : i32
      %c56_i64 = arith.constant 56 : i64
      %66 = arith.shrui %53, %c56_i64 : i64
      %67 = arith.trunci %66 : i64 to i32
      %c1_i32 = arith.constant 1 : i32
      %c7_i32 = arith.constant 7 : i32
      %68 = arith.shrui %67, %c7_i32 : i32
      %69 = arith.andi %68, %c1_i32 : i32
      %c255_i32_10 = arith.constant 255 : i32
      %c0_i32_11 = arith.constant 0 : i32
      %70 = arith.subi %c0_i32_11, %69 : i32
      %71 = arith.andi %70, %c255_i32_10 : i32
      %c24_i32_12 = arith.constant 24 : i32
      %72 = arith.shli %71, %c24_i32_12 : i32
      %73 = arith.ori %65, %72 : i32
      %c268435455_i32 = arith.constant 268435455 : i32
      %c4_i32 = arith.constant 4 : i32
      %74 = arith.shli %c268435455_i32, %c4_i32 : i32
      %c-1_i32 = arith.constant -1 : i32
      %75 = arith.xori %74, %c-1_i32 : i32
      %76 = arith.andi %arg3, %75 : i32
      %c4_i32_13 = arith.constant 4 : i32
      %77 = arith.shli %arg3, %c4_i32_13 : i32
      %78 = arith.andi %77, %74 : i32
      %79 = arith.ori %76, %78 : i32
      %c511_i32 = arith.constant 511 : i32
      %c23_i32 = arith.constant 23 : i32
      %80 = arith.shrui %arg3, %c23_i32 : i32
      %81 = arith.andi %80, %c511_i32 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %c31_i32 = arith.constant 31 : i32
      %82 = arith.shrui %arg3, %c31_i32 : i32
      %83 = arith.andi %82, %c1_i32_14 : i32
      %c9_i32 = arith.constant 9 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %84 = arith.subi %c0_i32_15, %83 : i32
      %85 = arith.shli %84, %c9_i32 : i32
      %86 = arith.ori %81, %85 : i32
      %c4_i32_16 = arith.constant 4 : i32
      %87 = arith.muli %11, %c4_i32_16 : i32
      %c0_i32_17 = arith.constant 0 : i32
      %88 = arith.addi %87, %c0_i32_17 : i32
      %89 = arith.index_cast %88 : i32 to index
      memref.store %79, %arg0[%89] : memref<256xi32>
      %c4_i32_18 = arith.constant 4 : i32
      %90 = arith.muli %11, %c4_i32_18 : i32
      %c1_i32_19 = arith.constant 1 : i32
      %91 = arith.addi %90, %c1_i32_19 : i32
      %92 = arith.index_cast %91 : i32 to index
      memref.store %86, %arg0[%92] : memref<256xi32>
      %c4_i32_20 = arith.constant 4 : i32
      %93 = arith.muli %11, %c4_i32_20 : i32
      %c2_i32 = arith.constant 2 : i32
      %94 = arith.addi %93, %c2_i32 : i32
      %95 = arith.index_cast %94 : i32 to index
      memref.store %49, %arg0[%95] : memref<256xi32>
      %c4_i32_21 = arith.constant 4 : i32
      %96 = arith.muli %11, %c4_i32_21 : i32
      %c3_i32 = arith.constant 3 : i32
      %97 = arith.addi %96, %c3_i32 : i32
      %98 = arith.index_cast %97 : i32 to index
      memref.store %73, %arg0[%98] : memref<256xi32>
    } else {
    }
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
    %v0 = arith.constant -1 : i32
    %k0 = arith.constant 0 : index
    memref.store %v0, %in[%k0] : memref<64xi32>
    %v1 = arith.constant 84853737 : i32
    %k1 = arith.constant 1 : index
    memref.store %v1, %in[%k1] : memref<64xi32>
    %v2 = arith.constant -1448634495 : i32
    %k2 = arith.constant 2 : index
    memref.store %v2, %in[%k2] : memref<64xi32>
    %v3 = arith.constant -437094558 : i32
    %k3 = arith.constant 3 : index
    memref.store %v3, %in[%k3] : memref<64xi32>
    %v4 = arith.constant 451858896 : i32
    %k4 = arith.constant 4 : index
    memref.store %v4, %in[%k4] : memref<64xi32>
    %v5 = arith.constant -2144010339 : i32
    %k5 = arith.constant 5 : index
    memref.store %v5, %in[%k5] : memref<64xi32>
    %v6 = arith.constant -1352809847 : i32
    %k6 = arith.constant 6 : index
    memref.store %v6, %in[%k6] : memref<64xi32>
    %v7 = arith.constant -1534019628 : i32
    %k7 = arith.constant 7 : index
    memref.store %v7, %in[%k7] : memref<64xi32>
    %v8 = arith.constant 0 : i32
    %k8 = arith.constant 8 : index
    memref.store %v8, %in[%k8] : memref<64xi32>
    %v9 = arith.constant -1126590959 : i32
    %k9 = arith.constant 9 : index
    memref.store %v9, %in[%k9] : memref<64xi32>
    %v10 = arith.constant -770361762 : i32
    %k10 = arith.constant 10 : index
    memref.store %v10, %in[%k10] : memref<64xi32>
    %v11 = arith.constant 1755676021 : i32
    %k11 = arith.constant 11 : index
    memref.store %v11, %in[%k11] : memref<64xi32>
    %v12 = arith.constant -1350363458 : i32
    %k12 = arith.constant 12 : index
    memref.store %v12, %in[%k12] : memref<64xi32>
    %v13 = arith.constant -2118736626 : i32
    %k13 = arith.constant 13 : index
    memref.store %v13, %in[%k13] : memref<64xi32>
    %v14 = arith.constant -265325687 : i32
    %k14 = arith.constant 14 : index
    memref.store %v14, %in[%k14] : memref<64xi32>
    %v15 = arith.constant 1775222487 : i32
    %k15 = arith.constant 15 : index
    memref.store %v15, %in[%k15] : memref<64xi32>
    %v16 = arith.constant -2147483648 : i32
    %k16 = arith.constant 16 : index
    memref.store %v16, %in[%k16] : memref<64xi32>
    %v17 = arith.constant -1163024424 : i32
    %k17 = arith.constant 17 : index
    memref.store %v17, %in[%k17] : memref<64xi32>
    %v18 = arith.constant 1865537624 : i32
    %k18 = arith.constant 18 : index
    memref.store %v18, %in[%k18] : memref<64xi32>
    %v19 = arith.constant 867440552 : i32
    %k19 = arith.constant 19 : index
    memref.store %v19, %in[%k19] : memref<64xi32>
    %v20 = arith.constant 384102704 : i32
    %k20 = arith.constant 20 : index
    memref.store %v20, %in[%k20] : memref<64xi32>
    %v21 = arith.constant -498595516 : i32
    %k21 = arith.constant 21 : index
    memref.store %v21, %in[%k21] : memref<64xi32>
    %v22 = arith.constant -944371144 : i32
    %k22 = arith.constant 22 : index
    memref.store %v22, %in[%k22] : memref<64xi32>
    %v23 = arith.constant -2059174781 : i32
    %k23 = arith.constant 23 : index
    memref.store %v23, %in[%k23] : memref<64xi32>
    %v24 = arith.constant -2147483648 : i32
    %k24 = arith.constant 24 : index
    memref.store %v24, %in[%k24] : memref<64xi32>
    %v25 = arith.constant 1220804323 : i32
    %k25 = arith.constant 25 : index
    memref.store %v25, %in[%k25] : memref<64xi32>
    %v26 = arith.constant -1211940493 : i32
    %k26 = arith.constant 26 : index
    memref.store %v26, %in[%k26] : memref<64xi32>
    %v27 = arith.constant -272476916 : i32
    %k27 = arith.constant 27 : index
    memref.store %v27, %in[%k27] : memref<64xi32>
    %v28 = arith.constant -1973268975 : i32
    %k28 = arith.constant 28 : index
    memref.store %v28, %in[%k28] : memref<64xi32>
    %v29 = arith.constant -589346422 : i32
    %k29 = arith.constant 29 : index
    memref.store %v29, %in[%k29] : memref<64xi32>
    %v30 = arith.constant 1154569014 : i32
    %k30 = arith.constant 30 : index
    memref.store %v30, %in[%k30] : memref<64xi32>
    %v31 = arith.constant 874816876 : i32
    %k31 = arith.constant 31 : index
    memref.store %v31, %in[%k31] : memref<64xi32>
    %v32 = arith.constant -2147483648 : i32
    %k32 = arith.constant 32 : index
    memref.store %v32, %in[%k32] : memref<64xi32>
    %v33 = arith.constant 1798509290 : i32
    %k33 = arith.constant 33 : index
    memref.store %v33, %in[%k33] : memref<64xi32>
    %v34 = arith.constant -1481734595 : i32
    %k34 = arith.constant 34 : index
    memref.store %v34, %in[%k34] : memref<64xi32>
    %v35 = arith.constant -283214773 : i32
    %k35 = arith.constant 35 : index
    memref.store %v35, %in[%k35] : memref<64xi32>
    %v36 = arith.constant -1131633585 : i32
    %k36 = arith.constant 36 : index
    memref.store %v36, %in[%k36] : memref<64xi32>
    %v37 = arith.constant 2029512098 : i32
    %k37 = arith.constant 37 : index
    memref.store %v37, %in[%k37] : memref<64xi32>
    %v38 = arith.constant 817434450 : i32
    %k38 = arith.constant 38 : index
    memref.store %v38, %in[%k38] : memref<64xi32>
    %v39 = arith.constant 52930877 : i32
    %k39 = arith.constant 39 : index
    memref.store %v39, %in[%k39] : memref<64xi32>
    %v40 = arith.constant 0 : i32
    %k40 = arith.constant 40 : index
    memref.store %v40, %in[%k40] : memref<64xi32>
    %v41 = arith.constant -1013074672 : i32
    %k41 = arith.constant 41 : index
    memref.store %v41, %in[%k41] : memref<64xi32>
    %v42 = arith.constant -1810977905 : i32
    %k42 = arith.constant 42 : index
    memref.store %v42, %in[%k42] : memref<64xi32>
    %v43 = arith.constant -495383377 : i32
    %k43 = arith.constant 43 : index
    memref.store %v43, %in[%k43] : memref<64xi32>
    %v44 = arith.constant 170000569 : i32
    %k44 = arith.constant 44 : index
    memref.store %v44, %in[%k44] : memref<64xi32>
    %v45 = arith.constant 1040951910 : i32
    %k45 = arith.constant 45 : index
    memref.store %v45, %in[%k45] : memref<64xi32>
    %v46 = arith.constant 1328681572 : i32
    %k46 = arith.constant 46 : index
    memref.store %v46, %in[%k46] : memref<64xi32>
    %v47 = arith.constant 849991431 : i32
    %k47 = arith.constant 47 : index
    memref.store %v47, %in[%k47] : memref<64xi32>
    %v48 = arith.constant -1 : i32
    %k48 = arith.constant 48 : index
    memref.store %v48, %in[%k48] : memref<64xi32>
    %v49 = arith.constant 1996438266 : i32
    %k49 = arith.constant 49 : index
    memref.store %v49, %in[%k49] : memref<64xi32>
    %v50 = arith.constant -885604635 : i32
    %k50 = arith.constant 50 : index
    memref.store %v50, %in[%k50] : memref<64xi32>
    %v51 = arith.constant 644528993 : i32
    %k51 = arith.constant 51 : index
    memref.store %v51, %in[%k51] : memref<64xi32>
    %v52 = arith.constant -1133062747 : i32
    %k52 = arith.constant 52 : index
    memref.store %v52, %in[%k52] : memref<64xi32>
    %v53 = arith.constant 2090271985 : i32
    %k53 = arith.constant 53 : index
    memref.store %v53, %in[%k53] : memref<64xi32>
    %v54 = arith.constant -157607487 : i32
    %k54 = arith.constant 54 : index
    memref.store %v54, %in[%k54] : memref<64xi32>
    %v55 = arith.constant 204844229 : i32
    %k55 = arith.constant 55 : index
    memref.store %v55, %in[%k55] : memref<64xi32>
    %v56 = arith.constant -2147483648 : i32
    %k56 = arith.constant 56 : index
    memref.store %v56, %in[%k56] : memref<64xi32>
    %v57 = arith.constant -786270723 : i32
    %k57 = arith.constant 57 : index
    memref.store %v57, %in[%k57] : memref<64xi32>
    %v58 = arith.constant -2070727766 : i32
    %k58 = arith.constant 58 : index
    memref.store %v58, %in[%k58] : memref<64xi32>
    %v59 = arith.constant 665251955 : i32
    %k59 = arith.constant 59 : index
    memref.store %v59, %in[%k59] : memref<64xi32>
    %v60 = arith.constant 1324617531 : i32
    %k60 = arith.constant 60 : index
    memref.store %v60, %in[%k60] : memref<64xi32>
    %v61 = arith.constant -2108585201 : i32
    %k61 = arith.constant 61 : index
    memref.store %v61, %in[%k61] : memref<64xi32>
    %v62 = arith.constant 1450817279 : i32
    %k62 = arith.constant 62 : index
    memref.store %v62, %in[%k62] : memref<64xi32>
    %v63 = arith.constant 1863553770 : i32
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
