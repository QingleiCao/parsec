include(${CMAKE_CURRENT_LIST_DIR}/ptgpp/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/user-defined-functions/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/branching/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/multisize_bcast/Testings.cmake)

parsec_addtest_cmd(dsl/ptg/startup1 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=10 -k=10 -v=5)
parsec_addtest_cmd(dsl/ptg/startup2 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=20 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/startup3 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=30 -j=30 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/strange ${SHM_TEST_CMD_LIST} dsl/ptg/strange)
if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_cmd(dsl/ptg/batch_cpu ${SHM_TEST_CMD_LIST} dsl/ptg/batch_cpu)
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)

if( MPI_C_FOUND )
  # Two flows of one task broadcast to different sets of ranks, under the star,
  # the chain and the binomial tree
  parsec_addtest_cmd(dsl/ptg/multiflow_bcast:mp:star     ${MPI_TEST_CMD_LIST} 4 dsl/ptg/multiflow_bcast -- --mca runtime_comm_coll_bcast 0)
  parsec_addtest_cmd(dsl/ptg/multiflow_bcast:mp:chain    ${MPI_TEST_CMD_LIST} 4 dsl/ptg/multiflow_bcast -- --mca runtime_comm_coll_bcast 1)
  parsec_addtest_cmd(dsl/ptg/multiflow_bcast:mp:binomial ${MPI_TEST_CMD_LIST} 4 dsl/ptg/multiflow_bcast -- --mca runtime_comm_coll_bcast 2)
endif( MPI_C_FOUND )
