include(runtime/scheduling/Testings.cmake)
include(runtime/cuda/Testings.cmake)

if( MPI_C_FOUND )
  parsec_addtest_cmd(runtime/many_activations:mp ${MPI_TEST_CMD_LIST} 2 runtime/many_activations -n 1000)
  set_tests_properties(runtime/many_activations:mp PROPERTIES TIMEOUT 120)
endif( MPI_C_FOUND )
