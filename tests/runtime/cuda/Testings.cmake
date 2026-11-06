if(PARSEC_HAVE_CUDA)
  # Every CUDA test below assumes the kernels this build produced can run on
  # the device it finds. When they cannot, none of them says so: the launches
  # fail, the kernels write nothing, and the tests report wrong results or
  # deadlock instead. Ask the question once, up front, and let the tests that
  # depend on the answer not run at all when it is no.
  if(TARGET cuda_arch_probe)
    parsec_addtest_cmd(runtime/cuda/arch_probe:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/cuda_arch_probe)
    set_tests_properties(runtime/cuda/arch_probe:gpu PROPERTIES FIXTURES_SETUP parsec_cuda_kernels_run)
  endif()
  parsec_addtest_cmd(runtime/cuda/get_best_device:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/testing_get_best_device -N 400 -t 20 -g 4 -- --mca device_show_statistics 1)

  # Each task handles one 32x32 double tile (8192 B). RW inputs and forced
  # descriptor outputs must both be required and transferred.
  set(device_stats_cuda_3 "[|]  Dev [ ]*[0-9]+ [|][ ]*3 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_5 "[|]  Dev [ ]*[0-9]+ [|][ ]*5 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_8 "[|]  Dev [ ]*[0-9]+ [|][ ]*8 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_10 "[|]  Dev [ ]*[0-9]+ [|][ ]*10 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario explicit)
  set_tests_properties(runtime/cuda/device_show_capabilities:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_3}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/default:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario default)
  set_tests_properties(runtime/cuda/device_show_capabilities/default:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_10}")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/default_matrix:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario default)
  set_tests_properties(runtime/cuda/device_show_capabilities/default_matrix:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/unfinished:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario unfinished)
  set_tests_properties(runtime/cuda/device_show_capabilities/unfinished:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/restart:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario restart)
  set_tests_properties(runtime/cuda/device_show_capabilities/restart:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/reset:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario reset)
  set_tests_properties(runtime/cuda/device_show_capabilities/reset:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/end_without_start:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_capabilities/end_without_start:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "parsec_device_show_capabilities_end called without a matching start")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/end_without_start_default:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_capabilities/end_without_start_default:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_3}")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/disabled:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario disabled)
  set_tests_properties(runtime/cuda/device_show_capabilities/disabled:gpu PROPERTIES
    FAIL_REGULAR_EXPRESSION "#[ ]KERNEL;Full transfer matrix")

  if(TARGET nvlink)
    parsec_addtest_cmd(runtime/cuda/nvlink:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/nvlink --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stress)
    parsec_addtest_cmd(runtime/cuda/stress:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -- --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    # Far less memory than the tasks need, so the device evicts and reuses
    # copies constantly. Nothing else in the suite makes it reclaim memory,
    # where a copy left with readers leaks and teardown reports it.
    parsec_addtest_cmd(runtime/cuda/stress:gpu:capped ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -n 512 -d 60 -- --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_show_statistics 1)
    if(PARSEC_DEBUG OR PARSEC_DEBUG_NOISIER)
      # device_inject_disable is debug-only. It sends a task back out through
      # parsec_device_kernel_cleanout(), the exit that skips the pop stage.
      parsec_addtest_cmd(runtime/cuda/stress:gpu:capped:disable ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -n 512 -d 60 -- --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_inject_disable 7 --mca device_show_statistics 1)
    endif()
  endif()
  if(TARGET stage)
    parsec_addtest_cmd(runtime/cuda/stage:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stage --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET cuda_rtt)
    parsec_addtest_cmd(runtime/cuda/rtt:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/cuda_rtt -g 1 -l 10 -m 4096 -- --mca device_cuda_enabled 1 --mca device_show_statistics 1)
  endif()
  if(TARGET ptg_pingpong)
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/ptg_pingpong:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
  if(TARGET dtd_pingpong)
    parsec_addtest_cmd(runtime/cuda/dtd_pingpong:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/dtd_pingpong --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/dtd_pingpong:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
  if(MPI_C_FOUND AND TARGET ptg_pingpong)
    # Eviction with remote dependencies: the comm thread reads copies the
    # device may be reclaiming, and a forwarded output is read by both the
    # receive and the send. Neither happens in a single-process run.
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:mp:gpu:capped ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/ptg_pingpong:mp:gpu:capped PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
endif()

if(PARSEC_HAVE_HIP)
  if(TARGET ptg_pingpong)
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET dtd_pingpong)
    parsec_addtest_cmd(runtime/cuda/dtd_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/dtd_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
endif()

if(PARSEC_HAVE_HIP)
  # Each task handles one 32x32 double tile (8192 B). RW inputs and forced
  # descriptor outputs must both be required and transferred.
  set(device_stats_hip_3 "[|]  Dev [ ]*[0-9]+ [|][ ]*3 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_5 "[|]  Dev [ ]*[0-9]+ [|][ ]*5 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_8 "[|]  Dev [ ]*[0-9]+ [|][ ]*8 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_10 "[|]  Dev [ ]*[0-9]+ [|][ ]*10 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario explicit)
  set_tests_properties(runtime/cuda/device_show_capabilities:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_3}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/default:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario default)
  set_tests_properties(runtime/cuda/device_show_capabilities/default:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_10}")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/default_matrix:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario default)
  set_tests_properties(runtime/cuda/device_show_capabilities/default_matrix:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/unfinished:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario unfinished)
  set_tests_properties(runtime/cuda/device_show_capabilities/unfinished:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/restart:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario restart)
  set_tests_properties(runtime/cuda/device_show_capabilities/restart:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/reset:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario reset)
  set_tests_properties(runtime/cuda/device_show_capabilities/reset:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/end_without_start:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_capabilities/end_without_start:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "parsec_device_show_capabilities_end called without a matching start")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/end_without_start_default:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_capabilities/end_without_start_default:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_3}")

  parsec_addtest_cmd(runtime/cuda/device_show_capabilities/disabled:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_capabilities --scenario disabled)
  set_tests_properties(runtime/cuda/device_show_capabilities/disabled:hip PROPERTIES
    FAIL_REGULAR_EXPRESSION "#[ ]KERNEL;Full transfer matrix")
endif()
