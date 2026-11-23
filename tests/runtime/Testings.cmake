#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(runtime/scheduling/Testings.cmake)
include(runtime/cuda/Testings.cmake)

if( MPI_C_FOUND )
  parsec_addtest_cmd(runtime/multichain:mp ${MPI_TEST_CMD_LIST} 4 runtime/multichain -l=1 -c=2)
endif( MPI_C_FOUND )

# The inputs of a host task whose class has a flow it only writes
parsec_addtest_cmd(runtime/host_inputs ${SHM_TEST_CMD_LIST} runtime/host_inputs)
if(PARSEC_HAVE_CUDA)
  parsec_addtest_cmd(runtime/host_inputs:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/host_inputs -g -- --mca device_cuda_enabled 1)
endif()
