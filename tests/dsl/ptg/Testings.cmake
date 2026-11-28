include(${CMAKE_CURRENT_LIST_DIR}/ptgpp/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/user-defined-functions/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/branching/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/multisize_bcast/Testings.cmake)

parsec_addtest_cmd(dsl/ptg/startup1 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=10 -k=10 -v=5)
parsec_addtest_cmd(dsl/ptg/startup2 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=20 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/startup3 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=30 -j=30 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/strange ${SHM_TEST_CMD_LIST} dsl/ptg/strange)
# A context has no QPU device unless one is asked for: the CPU body runs.
parsec_addtest_cmd(dsl/ptg/qpu_fallback ${SHM_TEST_CMD_LIST} dsl/ptg/qpu_fallback -x cpu)
if(PARSEC_HAVE_QPU)
  # With the device (the mock backend), the QPU body of the same task class runs.
  parsec_addtest_cmd(dsl/ptg/qpu_fallback:qpu ${SHM_TEST_CMD_LIST} dsl/ptg/qpu_fallback -x qpu
                     -- --mca device_qpu_enabled 1)
  set_tests_properties(dsl/ptg/qpu_fallback:qpu PROPERTIES TIMEOUT 180)
endif(PARSEC_HAVE_QPU)
if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_cmd(dsl/ptg/batch_cpu ${SHM_TEST_CMD_LIST} dsl/ptg/batch_cpu)
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
