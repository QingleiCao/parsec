#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# The quantum jobs go to the exact sampler of the QPU component or to its
# mock backend; no test can reach a device. The first four tests do not
# start the runtime; the others run the application on it.

if(PARSEC_HAVE_QPU)
  function(qaoa_addtest name)
    parsec_addtest_cmd(${name} ${ARGN})
    set_tests_properties(${name} PROPERTIES TIMEOUT 300)
  endfunction(qaoa_addtest)

  # The arithmetic: decomposition, fields, bit order, circuit, closed forms,
  # accept rule, optimizer, enumerations, statistics.
  qaoa_addtest(apps/qaoa/math ${SHM_TEST_CMD_LIST} apps/qaoa/qaoa_math_test)

  # The sequential reference on every fixed instance, with its invariants.
  qaoa_addtest(apps/qaoa/reference ${SHM_TEST_CMD_LIST} apps/qaoa/qaoa_reference)

  # The same program as it is run for a measurement: a generated graph, two
  # layers, timers; and every job a blocking call of one millisecond through
  # the mock backend, which is the sequential baseline.
  qaoa_addtest(apps/qaoa/reference:generated ${SHM_TEST_CMD_LIST} apps/qaoa/qaoa_reference
               -g 6:6:3:2 -p 2 -T 3 -R 2 -s 200 -P 1 -t)
  qaoa_addtest(apps/qaoa/reference:blocking ${SHM_TEST_CMD_LIST} apps/qaoa/qaoa_reference
               -i ring64 -b -d const:1000 -T 2 -R 2)

  # The application, one taskpool per instance and setting, each compared
  # with the sequential reference: the same states, byte for byte. A job
  # takes up to two milliseconds, drawn per job, so jobs end in an order
  # that is not the one they were submitted in.
  set(_qaoa -- --mca device_qpu_enabled 1 --mca device_qpu_mock_delay uniform:0:2000)
  qaoa_addtest(apps/qaoa ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa ${_qaoa})
  # Other seeds of the mock: other samples, and other job times.
  qaoa_addtest(apps/qaoa:seed2 ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa ${_qaoa} --mca device_qpu_mock_seed 2)
  qaoa_addtest(apps/qaoa:seed3 ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa ${_qaoa} --mca device_qpu_mock_seed 3)
  # One job at a time at the QPU, and sixteen.
  qaoa_addtest(apps/qaoa:1slot ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa
               ${_qaoa} --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1)
  qaoa_addtest(apps/qaoa:16slots ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa
               ${_qaoa} --mca device_qpu_max_outstanding 16 --mca device_qpu_mock_slots 16 --mca device_qpu_mock_threads 4)
  # One worker thread: nothing in the application waits for a job.
  qaoa_addtest(apps/qaoa:1worker ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa ${_qaoa} --mca runtime_num_cores 1)
  # One evaluation in five fails for good, the same ones in the reference;
  # and every evaluation fails once and works when the runtime tries again,
  # which the application must not see.
  qaoa_addtest(apps/qaoa:fail ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -F keys ${_qaoa})
  qaoa_addtest(apps/qaoa:fail:retried ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -F first
               ${_qaoa} --mca device_qpu_mock_faults fail:first=1)
  # Tasks that share jobs: one job at a time at the QPU, of up to eight
  # tasks. The states are the reference's all the same; with evaluations
  # that fail for good, which take those that share their job down once;
  # and with every first job failing, shared or not.
  set(_qaoa_one ${_qaoa} --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1 --mca device_qpu_max_batch 8
                --mca device_qpu_mock_job_delay const:500)
  qaoa_addtest(apps/qaoa:coalesce ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -C ${_qaoa_one})
  qaoa_addtest(apps/qaoa:coalesce:fail ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -C -F keys ${_qaoa_one})
  qaoa_addtest(apps/qaoa:coalesce:fail:retried ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -C -F first
               ${_qaoa_one} --mca device_qpu_mock_faults fail:first=1)
  # Three places at the QPU and jobs of up to four tasks; and a minimum time between two submissions.
  qaoa_addtest(apps/qaoa:coalesce:3slots ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -C
               ${_qaoa} --mca device_qpu_max_outstanding 3 --mca device_qpu_mock_slots 3 --mca device_qpu_max_batch 4)
  qaoa_addtest(apps/qaoa:coalesce:interval ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -C -i ring64
               ${_qaoa} --mca device_qpu_max_batch 16 --mca device_qpu_submit_interval 2)
  # The colour-synchronous control (C-sync): every instance under four
  # settings, the same states, and no block of a colour started before the
  # colour before was done.
  qaoa_addtest(apps/qaoa:sync ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -y ${_qaoa})
  qaoa_addtest(apps/qaoa:sync:fail ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -y -F keys ${_qaoa})
  # Block graphs of other shapes: a grid, all pairs, random pairs; with C-sync on the random one.
  qaoa_addtest(apps/qaoa:grid ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -g grid:16:6:2:2 -T 3 -R 3 ${_qaoa})
  qaoa_addtest(apps/qaoa:complete ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -g complete:8:5:2:1 -T 3 -R 3 ${_qaoa})
  qaoa_addtest(apps/qaoa:rand ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -g rand:12:6:2:2 -T 3 -R 6 -P 1 ${_qaoa})
  qaoa_addtest(apps/qaoa:rand:sync ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -y -g rand:12:6:2:2 -T 3 -R 6 -P 1 ${_qaoa})
  # A generated instance, as for a measurement: 8 blocks of 10 qubits, two layers.
  qaoa_addtest(apps/qaoa:generated ${SHM_TEST_CMD_LIST} apps/qaoa/testing_qaoa -g 8:10:4:3 -p 2 -T 4 -R 3 -s 256 ${_qaoa})

  if( MPI_C_FOUND )
    # The same on several ranks: block b on rank b modulo their number, the
    # QPU on rank 0. Some instances have fewer blocks than there are ranks.
    foreach(_np 2 4)
      qaoa_addtest(apps/qaoa:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} apps/qaoa/testing_qaoa ${_qaoa})
    endforeach()
    # Three ranks: a block count that is a multiple of the rank count for some instances only.
    qaoa_addtest(apps/qaoa:mp:3 ${MPI_TEST_CMD_LIST} 3 apps/qaoa/testing_qaoa ${_qaoa})
    # The QPU on the last rank.
    qaoa_addtest(apps/qaoa:mp:last ${MPI_TEST_CMD_LIST} 4 apps/qaoa/testing_qaoa ${_qaoa} --mca device_qpu_owner_rank -1)
    qaoa_addtest(apps/qaoa:fail:mp ${MPI_TEST_CMD_LIST} 2 apps/qaoa/testing_qaoa -F keys ${_qaoa})
    # C-sync on three ranks, and with failing evaluations on four.
    qaoa_addtest(apps/qaoa:sync:mp:3 ${MPI_TEST_CMD_LIST} 3 apps/qaoa/testing_qaoa -y ${_qaoa})
    qaoa_addtest(apps/qaoa:sync:fail:mp ${MPI_TEST_CMD_LIST} 4 apps/qaoa/testing_qaoa -y -F keys ${_qaoa})
    # Baseline B, the hand-written MPI program, on 1, 2 and 4 ranks; on a
    # random block graph with skipped updates; with one job at a time.
    foreach(_np 1 2 4)
      qaoa_addtest(apps/qaoa/mpi:${_np} ${MPI_TEST_CMD_LIST} ${_np} apps/qaoa/qaoa_mpi ${_qaoa})
    endforeach()
    qaoa_addtest(apps/qaoa/mpi:rand ${MPI_TEST_CMD_LIST} 3 apps/qaoa/qaoa_mpi -g rand:12:6:2:2 -T 3 -R 6 -P 1 ${_qaoa})
    # More ranks than blocks: a rank with no block takes part in every exchange.
    qaoa_addtest(apps/qaoa/mpi:spare ${MPI_TEST_CMD_LIST} 4 apps/qaoa/qaoa_mpi -i rand12 ${_qaoa})
    # C-sync with more ranks than blocks, and a complete block graph (a colour per block).
    qaoa_addtest(apps/qaoa:sync:spare ${MPI_TEST_CMD_LIST} 4 apps/qaoa/testing_qaoa -y -i rand12 ${_qaoa})
    qaoa_addtest(apps/qaoa:sync:complete ${MPI_TEST_CMD_LIST} 3 apps/qaoa/testing_qaoa -y -g complete:8:5:2:1 -T 2 -R 2 ${_qaoa})
    qaoa_addtest(apps/qaoa/mpi:1slot ${MPI_TEST_CMD_LIST} 2 apps/qaoa/qaoa_mpi -g grid:16:6:2:2 -T 3 -R 2
                 ${_qaoa} --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1)
    # Tasks that share jobs, the blocks on four ranks; and with failing evaluations, the QPU on the last of three.
    qaoa_addtest(apps/qaoa:coalesce:mp ${MPI_TEST_CMD_LIST} 4 apps/qaoa/testing_qaoa -C ${_qaoa_one})
    qaoa_addtest(apps/qaoa:coalesce:fail:mp ${MPI_TEST_CMD_LIST} 3 apps/qaoa/testing_qaoa -C -F keys
                 ${_qaoa_one} --mca device_qpu_owner_rank -1)
  endif( MPI_C_FOUND )
endif(PARSEC_HAVE_QPU)
