if(PARSEC_HAVE_CUDA)
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
    parsec_addtest_cmd(runtime/cuda/stress:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stage)
    parsec_addtest_cmd(runtime/cuda/stage:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stage --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET alloc_size)
    parsec_addtest_cmd(runtime/cuda/alloc_size:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/alloc_size --mca device_cuda_enabled 1 --mca device_show_statistics 1)
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
