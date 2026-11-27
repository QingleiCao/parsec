#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Every test here opts in to the QPU layer with device_qpu_enabled and runs on
# the mock backend, or on a backend of its own that drives nothing. Delays are
# kept at or below 100 ms, except where a test is about a job that outlasts
# something else.

if(PARSEC_HAVE_QPU)
  # Register a QPU test. A completion that never comes is a hang, so every
  # one of them has a timeout.
  function(qpu_addtest name)
    parsec_addtest_cmd(${name} ${ARGN})
    set_tests_properties(${name} PROPERTIES TIMEOUT 180)
  endfunction(qpu_addtest)

  set(_qpu -- --mca device_qpu_enabled 1)

  add_test(NAME runtime/qpu/no_hardware_in_tests
           COMMAND ${CMAKE_COMMAND} -DTESTS_DIR=${PROJECT_SOURCE_DIR}/tests
                   -P ${PROJECT_SOURCE_DIR}/tests/runtime/qpu/check_no_hardware.cmake)

  # A backend that drives, or says it drives, a quantum device is refused;
  # one that does not exist is not replaced by another; and the same task
  # runs when the mock is asked for.
  qpu_addtest(runtime/qpu/hardware_guard ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend guard_physical)
  qpu_addtest(runtime/qpu/hardware_guard:late ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend guard_late)
  qpu_addtest(runtime/qpu/hardware_guard:unknown ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend guard_missing)
  qpu_addtest(runtime/qpu/hardware_guard:control ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend mock)
  # In a list of QPUs, one backend that may not be used takes the others with
  # it: the mock that is listed first is not opened either.
  qpu_addtest(runtime/qpu/hardware_guard:list ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend guard_physical --mca device_qpu_devices "mock|guard_physical")
  qpu_addtest(runtime/qpu/hardware_guard:list:unknown ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_guard
              ${_qpu} --mca device_qpu_backend guard_missing --mca device_qpu_devices "mock|guard_missing")

  # The QPU as a device: none unless asked for; one on every rank when it is;
  # a backend on the rank that owns it and nowhere else; and a quantum task on
  # any other rank is refused.
  qpu_addtest(runtime/qpu/discovery ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_discovery ${_qpu})
  qpu_addtest(runtime/qpu/discovery:off ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_discovery -x)
  # No task at all: the backend that was opened with the device is closed unused.
  qpu_addtest(runtime/qpu/discovery:idle ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_discovery -i ${_qpu})
  qpu_addtest(runtime/qpu/discovery:noowner ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_discovery -o -2
              ${_qpu} --mca device_qpu_owner_rank 5)

  # CPU, GPU and QPU incarnations in one task graph over the same tiles: what
  # an accelerator wrote is read by a quantum task, and what the quantum task
  # wrote is read by an accelerator. Without a CUDA device the CPU bodies of
  # the same task classes stand in.
  qpu_addtest(runtime/qpu/cpu_gpu_qpu ${SHM_TEST_CMD_LIST} runtime/qpu/cpu_gpu_qpu -n 16 -q 1
              ${_qpu} --mca device_qpu_mock_delay uniform:200:5000)
  # The same with three QPUs: the quantum task of a chain runs on whichever
  # has a place, between the same two accelerator tasks.
  qpu_addtest(runtime/qpu/cpu_gpu_qpu:qpus ${SHM_TEST_CMD_LIST} runtime/qpu/cpu_gpu_qpu -n 16 -q 3
              ${_qpu} --mca device_qpu_mock_delay uniform:200:5000 --mca device_qpu_devices "mock|mock|mock")
  # What a QPU prints with device_show_statistics: the tasks it ran. The
  # device layer prints its table first and sets its counts back to zero, so
  # the QPU has to have counted them itself.
  # The run has to succeed and to print the line: repeat_test.cmake checks both.
  add_test(NAME runtime/qpu/cpu_gpu_qpu:statistics
           COMMAND ${CMAKE_COMMAND} -DREPEAT=1 "-DEXPECT=QPU 0, qpu[(]mock[)] on rank 0: 16 task[(]s[)], 16 job[(]s[)] at the backend"
                   -P ${PROJECT_SOURCE_DIR}/tests/runtime/qpu/repeat_test.cmake --
                   ${PROJECT_BINARY_DIR}/tests/runtime/qpu/cpu_gpu_qpu -n 16 -q 1
                   -- --mca device_qpu_enabled 1 --mca device_show_statistics 1)
  set_tests_properties(runtime/qpu/cpu_gpu_qpu:statistics PROPERTIES TIMEOUT 180)
  # The same line of each of two QPUs, with every block naming one in turn:
  # 12 of the 24 tasks each, whichever thread gets to them first.
  add_test(NAME runtime/qpu/multi:named:statistics
           COMMAND ${CMAKE_COMMAND} -DREPEAT=1
                   "-DEXPECT=QPU 0, qpu0[(]mock[)] on rank 0: 12 task[(]s[)], 12 job[(]s[)] at the backend.*QPU 1, qpu1[(]mock[)] on rank 0: 12 task[(]s[)], 12 job[(]s[)] at the backend"
                   -P ${PROJECT_SOURCE_DIR}/tests/runtime/qpu/repeat_test.cmake --
                   ${PROJECT_BINARY_DIR}/tests/runtime/qpu/qpu_multi -n 12 -l 2 -P turn
                   -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2
                   --mca device_qpu_mock_delay uniform:200:5000 --mca device_show_statistics 1
                   --mca device_qpu_devices "mock|mock")
  set_tests_properties(runtime/qpu/multi:named:statistics PROPERTIES TIMEOUT 180)
  if(PARSEC_HAVE_CUDA)
    qpu_addtest(runtime/qpu/cpu_gpu_qpu:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/qpu/cpu_gpu_qpu -g -n 16 -q 1
                ${_qpu} --mca device_cuda_enabled 1 --mca device_qpu_mock_delay uniform:200:5000)
    qpu_addtest(runtime/qpu/cpu_gpu_qpu:gpu:qpus ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/qpu/cpu_gpu_qpu -g -n 16 -q 3
                ${_qpu} --mca device_cuda_enabled 1 --mca device_qpu_mock_delay uniform:200:5000
                --mca device_qpu_devices "mock|mock|mock")
    # Two device components that both give modules: every device of both is there.
    qpu_addtest(runtime/qpu/discovery:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/qpu/qpu_discovery -g 1
                ${_qpu} --mca device_cuda_enabled 1)
  endif(PARSEC_HAVE_CUDA)

  # What a body may not submit: each case is refused, and the run goes on.
  qpu_addtest(runtime/qpu/misuse ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_misuse ${_qpu})

  # The sampler and the mock backend, without the runtime.
  qpu_addtest(runtime/qpu/sampler ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_sampler_test)
  qpu_addtest(runtime/qpu/mock ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_mock_test)

  # Submission, asynchronous completion, dependency release. Every fifth job
  # is empty.
  qpu_addtest(runtime/qpu/chain ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 32 -l 3 -e 5
              ${_qpu} --mca device_qpu_mock_delay uniform:200:5000)
  qpu_addtest(runtime/qpu/chain:dtd ${SHM_TEST_CMD_LIST} runtime/qpu/dtd_qpu_chain -n 32
              ${_qpu} --mca device_qpu_mock_delay const:20000)
  # The same chains with a quantum task whose body is not of the QPU type.
  qpu_addtest(runtime/qpu/chain:cputype ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain_cputype -n 32 -l 3 -e 5
              ${_qpu} --mca device_qpu_mock_delay uniform:200:5000)

  # Completion order: the dependents follow the order the jobs finish in.
  qpu_addtest(runtime/qpu/order:reverse ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 12 -l 1 -O reverse
              ${_qpu} --mca device_qpu_max_outstanding 12 --mca device_qpu_mock_slots 12)
  qpu_addtest(runtime/qpu/order:random ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 12 -l 1 -O random
              ${_qpu} --mca device_qpu_max_outstanding 12 --mca device_qpu_mock_slots 12)

  # Several outstanding tasks under a limit: far more tasks than the limit,
  # which has to be reached and never passed.
  qpu_addtest(runtime/qpu/capacity ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 64 -l 1 -c 6
              ${_qpu} --mca device_qpu_max_outstanding 6 --mca device_qpu_mock_slots 3
              --mca device_qpu_mock_delay const:20000)

  # The same with a minimum time between two submissions: the limit is still
  # reached, and no two jobs get to the backend closer than that. The jobs are
  # long next to the interval, so that the fourth is in before the first ends
  # even when the progress thread is kept waiting for a core.
  qpu_addtest(runtime/qpu/capacity:interval ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 24 -l 1 -c 4
              ${_qpu} --mca device_qpu_max_outstanding 4 --mca device_qpu_mock_slots 4
              --mca device_qpu_mock_delay const:60000 --mca device_qpu_submit_interval 5)

  # Classical work while a job is outstanding, with all workers and with one.
  qpu_addtest(runtime/qpu/overlap ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_overlap -d 400000
              ${_qpu} --mca device_qpu_mock_delay const:400000)
  qpu_addtest(runtime/qpu/overlap:1worker ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_overlap -t 1 -d 400000
              ${_qpu} --mca device_qpu_mock_delay const:400000)
  # These two measure a run against the length of its job. Every process binds
  # its first worker to the first core, so a test running beside them takes
  # time from the one worker: they run alone.
  set_tests_properties(runtime/qpu/overlap runtime/qpu/overlap:1worker PROPERTIES RUN_SERIAL TRUE)

  # Termination with tasks outstanding.
  qpu_addtest(runtime/qpu/termination ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_termination -n 8 -d 100000
              ${_qpu} --mca device_qpu_mock_delay const:100000)
  qpu_addtest(runtime/qpu/termination:two ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_termination -2 -n 8 -d 300000
              ${_qpu} --mca device_qpu_mock_delay const:300000)
  qpu_addtest(runtime/qpu/termination:loop ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_termination -L 50 -n 6 -d 2000
              ${_qpu} --mca device_qpu_mock_delay const:2000)
  if( MPI_C_FOUND )
    qpu_addtest(runtime/qpu/termination:dynamic ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_termination_dynamic -n 8 -d 100000
                ${_qpu} --mca device_qpu_mock_delay const:100000)
  endif( MPI_C_FOUND )

  # Races between submission, completion and sleep: many tasks, no delay or
  # next to none, with all workers, with one, and with two virtual processes.
  qpu_addtest(runtime/qpu/stress ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 400 -l 5
              ${_qpu} --mca device_qpu_mock_delay const:0 --mca device_qpu_max_outstanding 16 --mca device_qpu_mock_slots 16)
  qpu_addtest(runtime/qpu/stress:jitter ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 400 -l 5
              ${_qpu} --mca device_qpu_mock_delay uniform:0:300)
  qpu_addtest(runtime/qpu/stress:1worker ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 400 -l 5
              ${_qpu} --mca device_qpu_mock_delay const:0 --mca runtime_num_cores 1)
  # One virtual process per socket: more than one where there are several sockets.
  qpu_addtest(runtime/qpu/stress:vp ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -n 400 -l 5
              ${_qpu} --mca device_qpu_mock_delay uniform:0:300 --mca runtime_vpmap hwloc)

  # Jobs that fail, hang, are refused, lose their reply or report their end
  # twice; a backend that is down for a while, or for good; circuits the
  # backend cannot run. As many slots as jobs outstanding, so that a job that
  # hangs never keeps another one waiting past the time-out.
  set(_qpu_fail ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_mock_slots 8
                --mca device_qpu_job_timeout 250 --mca device_qpu_retry_backoff 2)
  qpu_addtest(runtime/qpu/failure ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 40 -m mixed ${_qpu_fail})
  qpu_addtest(runtime/qpu/failure:noretry ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 20 -m mixed
              ${_qpu_fail} --mca device_qpu_max_retries 0)
  qpu_addtest(runtime/qpu/failure:once ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m once ${_qpu_fail})
  qpu_addtest(runtime/qpu/failure:always ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m always ${_qpu_fail})
  qpu_addtest(runtime/qpu/failure:timeout ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 8 -m timeout ${_qpu_fail})
  qpu_addtest(runtime/qpu/failure:ambiguous ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 24 -m ambiguous ${_qpu_fail})
  qpu_addtest(runtime/qpu/failure:down:short ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m down:short
              ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_unavailable_timeout 5000
              --mca device_qpu_retry_backoff 5)
  qpu_addtest(runtime/qpu/failure:down:long ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m down:long
              ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_unavailable_timeout 200
              --mca device_qpu_retry_backoff 5)
  qpu_addtest(runtime/qpu/failure:rejected ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m rejected
              ${_qpu} --mca device_qpu_mock_max_qubits 3)
  # An outage that lasted past the time-out, then a second taskpool that is
  # refused three times and has to end well all the same.
  qpu_addtest(runtime/qpu/failure:down:twice ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m down:twice
              ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_unavailable_timeout 200
              --mca device_qpu_retry_backoff 5)
  # The backend takes a job and never says so: the job has to be gone from
  # it before the task that owns its inputs completes.
  qpu_addtest(runtime/qpu/failure:lost:down ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 16 -m lost:down
              ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_unavailable_timeout 200
              --mca device_qpu_retry_backoff 5)
  # Every kind of answer to a submission, with a minimum time between two
  # submissions: it holds whatever the backend answered.
  qpu_addtest(runtime/qpu/failure:interval ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_failure -n 20 -m mixed
              ${_qpu_fail} --mca device_qpu_submit_interval 3)

  # Tasks whose circuits share a job. One job at a time at the backend: the
  # tasks that wait go four by four.
  set(_qpu_one -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1
               --mca device_qpu_mock_job_delay const:2000)
  qpu_addtest(runtime/qpu/coalesce ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # The same with tasks that have no priority: they are taken as they arrived.
  qpu_addtest(runtime/qpu/coalesce:arrival ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -p 0
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # The same with no sharing allowed, and with a backend that takes two circuits per job.
  qpu_addtest(runtime/qpu/coalesce:off ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 ${_qpu_one})
  qpu_addtest(runtime/qpu/coalesce:cap ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17
              ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_mock_max_circuits 2)
  # A place at the backend for every task: nobody shares, whatever is allowed.
  qpu_addtest(runtime/qpu/coalesce:spare ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 6
              ${_qpu} --mca device_qpu_max_outstanding 8 --mca device_qpu_mock_slots 8 --mca device_qpu_max_batch 8)
  # Two places for thirteen tasks: seven and six, not eight and five.
  qpu_addtest(runtime/qpu/coalesce:spread ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 14
              ${_qpu} --mca device_qpu_max_outstanding 3 --mca device_qpu_mock_slots 3 --mca device_qpu_max_batch 8)
  # Places to spare and a minimum time between two submissions: each submission takes all it may.
  qpu_addtest(runtime/qpu/coalesce:interval ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 9
              ${_qpu} --mca device_qpu_max_outstanding 8 --mca device_qpu_mock_slots 8 --mca device_qpu_max_batch 4
              --mca device_qpu_submit_interval 5)
  # A job of several tasks that fails: each of them is retried alone, and
  # the one whose circuit always fails takes nobody with it a second time.
  qpu_addtest(runtime/qpu/coalesce:fail ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -f 8
              ${_qpu_one} --mca device_qpu_max_batch 4)
  qpu_addtest(runtime/qpu/coalesce:fail:always ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -F 8
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # Two tasks of a higher priority arrive while the job that fails is at the
  # backend: they go next, and the tasks that failed are not taken along.
  qpu_addtest(runtime/qpu/coalesce:fail:overtaken ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 19 -f 8 -w 2
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # ... and with no retry allowed, each of them completes as failed.
  qpu_addtest(runtime/qpu/coalesce:fail:noretry ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -f 8
              ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_max_retries 0)
  # Tasks that ask for different numbers of shots do not share a job: a job
  # ends where the number changes, and nobody is taken from further back.
  qpu_addtest(runtime/qpu/coalesce:shots ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 33 -a
              ${_qpu_one} --mca device_qpu_max_batch 8)
  # A job that somebody at the provider cancels: its tasks say so, and are not tried again.
  qpu_addtest(runtime/qpu/coalesce:cancelled ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -c 8
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # The time a job may take is that of a job of one task for each task it
  # carries: jobs of four circuits of 60 ms end well under a limit of 200 ms.
  qpu_addtest(runtime/qpu/coalesce:deadline ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 9 -d 60000
              ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_job_timeout 200)
  # Many tasks that arrive as they come, on two places.
  qpu_addtest(runtime/qpu/coalesce:stress ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 600 -m free -d 100
              ${_qpu} --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2 --mca device_qpu_max_batch 16
              --mca device_qpu_mock_job_delay const:300)

  # Jobs of several tasks through everything that can happen to a job: a
  # backend that takes every other submission, answers that are lost, jobs that are
  # refused, that end with no result, that are cancelled at the provider,
  # that hang until they are cancelled here, and ends reported twice. Two
  # places at the backend, jobs of up to eight tasks. The kinds that are
  # drawn per circuit are also given one evaluation by its key (that of
  # task k is k * 2^32), so that each of them is sure to happen.
  set(_qpu_two -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2
               --mca device_qpu_max_batch 8 --mca device_qpu_mock_job_delay const:300 --mca device_qpu_retry_backoff 2)
  qpu_addtest(runtime/qpu/coalesce:down ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 300 -x -d 100
              ${_qpu_two} --mca device_qpu_mock_faults unavailable:every=2)
  qpu_addtest(runtime/qpu/coalesce:lost ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 300 -x -d 100
              ${_qpu_two} --mca device_qpu_mock_faults lost:every=3)
  qpu_addtest(runtime/qpu/coalesce:mixed ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 300 -x -d 100
              ${_qpu_two} --mca device_qpu_mock_faults cancelled:key=644245094400,duplicate:key=648540061696,reject:every=7,noresult:p=0.03,fail:p=0.03,duplicate:p=0.05,unavailable:every=5,cancelled:p=0.02)
  qpu_addtest(runtime/qpu/coalesce:timeout ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 120 -x -d 100
              ${_qpu_two} --mca device_qpu_mock_faults hang:key=214748364800,late:key=219043332096,hang:p=0.02,late:p=0.02 --mca device_qpu_job_timeout 30)
  # The backend takes one job and is out of reach from then on, for longer
  # than the service waits: the tasks of the job that was being asked for,
  # and those that wait, complete as unavailable, and that job is taken back.
  qpu_addtest(runtime/qpu/coalesce:unavailable ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -u
              ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_unavailable_timeout 60 --mca device_qpu_retry_backoff 2)

  # The service stopped with a job of four tasks at the backend and twelve
  # tasks waiting: sixteen tasks given up, none completed.
  qpu_addtest(runtime/qpu/coalesce:abandoned ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -A -d 100000
              ${_qpu_one} --mca device_qpu_max_batch 4)
  # The same while the job of four tasks is being asked for and its answer is lost.
  qpu_addtest(runtime/qpu/coalesce:abandoned:asked ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_coalesce -n 17 -A -u
              ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_retry_backoff 2)
  # Fifty taskpools one after the other whose tasks are completed four by four.
  qpu_addtest(runtime/qpu/termination:loop:coalesce ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_termination -L 50 -n 9 -d 2000
              ${_qpu} --mca device_qpu_mock_delay const:500 --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1
              --mca device_qpu_max_batch 4)

  # The benchmark of experiments E1 and E2, small: the QPU body with jobs of
  # no time and of 2 ms; the same graph with an empty CPU body; a CPU body
  # that holds its worker for 2 ms; and the same work without the runtime.
  qpu_addtest(runtime/qpu/bench ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 8 -l 50 -w 20
              ${_qpu} --mca device_qpu_mock_delay const:0)
  qpu_addtest(runtime/qpu/bench:latency ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 16 -l 10
              ${_qpu} --mca device_qpu_mock_delay const:2000 --mca device_qpu_max_outstanding 16 --mca device_qpu_mock_slots 16
              --mca runtime_num_cores 2)
  qpu_addtest(runtime/qpu/bench:cpu ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 8 -l 50 -w 20)
  qpu_addtest(runtime/qpu/bench:block ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 16 -l 10 -b 2000 -- --mca runtime_num_cores 2)
  qpu_addtest(runtime/qpu/bench:sequential ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 4 -l 10 -b 2000 -A)
  # Work of 2 ms per link, longer than anything else in the run: a run that
  # does not do it ends before the least time its work can take.
  qpu_addtest(runtime/qpu/bench:work ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_bench -n 4 -l 10 -w 2000
              ${_qpu} --mca device_qpu_mock_delay const:0)

  # The QPU service started and stopped many times in one process.
  qpu_addtest(runtime/qpu/restart ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_chain -I 30 -n 8 -l 2
              ${_qpu} --mca device_qpu_mock_delay uniform:0:500)

  # Several QPUs in one run. Their list separates its entries with '|' here:
  # a ';' would end the argument in a CMake list.
  #
  # Two and three QPUs of one rank that are alike: every answer is the one of
  # its inputs whichever QPU gave it, no QPU goes over its own limit, and at
  # the end each task was counted by one QPU and no QPU carries a load. Then
  # the same with tasks that name their QPU, all of them and every other
  # chain; with jobs that carry several tasks; and with the list left out,
  # where device_qpu_enabled says how many QPUs there are.
  set(_qpu_m0 -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2)
  set(_qpu_m ${_qpu_m0} --mca device_qpu_mock_delay uniform:200:5000)
  qpu_addtest(runtime/qpu/multi ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 3
              ${_qpu_m} --mca device_qpu_devices "mock|mock")
  qpu_addtest(runtime/qpu/multi:three ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 3
              ${_qpu_m} --mca device_qpu_devices "mock | mock | mock")
  qpu_addtest(runtime/qpu/multi:enabled ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 3
              -- --mca device_qpu_enabled 3 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2
              --mca device_qpu_mock_delay uniform:200:5000)
  qpu_addtest(runtime/qpu/multi:named ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 3 -P turn
              ${_qpu_m} --mca device_qpu_devices "mock|mock|mock")
  qpu_addtest(runtime/qpu/multi:named:half ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 3 -P half
              ${_qpu_m} --mca device_qpu_devices "mock|mock|mock")
  qpu_addtest(runtime/qpu/multi:named:one ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 12 -l 2 -P 1
              ${_qpu_m} --mca device_qpu_devices "mock|mock|mock")
  qpu_addtest(runtime/qpu/multi:batch ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 60 -l 2
              ${_qpu_m} --mca device_qpu_devices "mock max_outstanding=1 slots=1|mock max_outstanding=1 slots=1"
              --mca device_qpu_max_batch 4 --mca device_qpu_mock_job_delay const:2000)
  # Each QPU has its own limits: one place here, three there.
  qpu_addtest(runtime/qpu/multi:limits ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 40 -l 2 -c
              ${_qpu_m0} --mca device_qpu_devices "mock max_outstanding=1 slots=1|mock max_outstanding=3 slots=3"
              --mca device_qpu_mock_delay const:10000)
  # No place stays free while a task waits: 48 tasks at once on three QPUs of
  # two places, with jobs of very different lengths.
  qpu_addtest(runtime/qpu/multi:noidle ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 48 -l 1 -W
              ${_qpu_m0} --mca device_qpu_devices "mock|mock|mock" --mca device_qpu_mock_delay exp:40000)
  # It reads times too, and runs alone: when other processes take its cores,
  # the three threads get to ask for their first jobs 10 to 30 ms after the
  # tasks were submitted, and those are moments with a free place. With the
  # same processes on other cores, not one moment is off.
  set_tests_properties(runtime/qpu/multi:noidle PROPERTIES RUN_SERIAL TRUE)
  # A task that is submitted while its QPU is at work, and that has a place the
  # moment after, is not left on the way: the thread looks once more before
  # it sleeps. Here the mock holds the thread, asked again for a job it did
  # not take at first, until the second task is submitted; that task found
  # no place shown and woke nobody. Jobs of two seconds, and threads that
  # sleep for 30 s when nobody wakes them.
  qpu_addtest(runtime/qpu/multi:wake:reask ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -G
              -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2
              --mca device_qpu_mock_delay const:2000000 --mca device_qpu_retry_backoff 2 --mca device_qpu_poll_max 30000)
  # A QPU that is twice as fast runs more than half of the tasks.
  qpu_addtest(runtime/qpu/multi:speeds ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 60 -l 1 -M 1
              ${_qpu_m} --mca device_qpu_devices "mock delay=const:20000|mock delay=const:10000")
  # Circuits of four qubits, and one QPU that is too narrow for them: the
  # other one runs them all. When both are too narrow, every task is rejected.
  qpu_addtest(runtime/qpu/multi:width ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -O 1
              ${_qpu_m} --mca device_qpu_devices "mock max_qubits=3|mock max_qubits=12")
  qpu_addtest(runtime/qpu/multi:width:none ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -R
              ${_qpu_m} --mca device_qpu_devices "mock max_qubits=3|mock max_qubits=2")
  # A QPU whose jobs all fail: a task that named none gets its answer from
  # the other QPU, and a task that named the one that fails ends without one.
  qpu_addtest(runtime/qpu/multi:fail ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -f 0
              ${_qpu_m} --mca device_qpu_devices "mock faults=fail:p=1|mock" --mca device_qpu_retry_backoff 2)
  qpu_addtest(runtime/qpu/multi:fail:named ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -f 0 -P half
              ${_qpu_m} --mca device_qpu_devices "mock faults=fail:p=1|mock" --mca device_qpu_retry_backoff 2)
  # The task of a job that failed goes to the other QPU at once, also when
  # that one sleeps and nothing else is going on: two chains only, and
  # threads that sleep for 30 s when nobody wakes them. The run is done in a
  # fraction of a second (it is given 5). Whether the last task of a run goes
  # to the QPU that fails first is a matter of turns, so the run is done 10
  # times.
  add_test(NAME runtime/qpu/multi:fail:prompt
           COMMAND ${CMAKE_COMMAND} -DREPEAT=10 -P ${PROJECT_SOURCE_DIR}/tests/runtime/qpu/repeat_test.cmake --
                   ${PROJECT_BINARY_DIR}/tests/runtime/qpu/qpu_multi -n 2 -l 6 -f 0 -t 5000
                   -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 2 --mca device_qpu_mock_slots 2
                   --mca device_qpu_mock_delay uniform:200:5000 --mca device_qpu_retry_backoff 2
                   --mca device_qpu_poll_max 30000 --mca device_qpu_devices "mock faults=fail:p=1|mock")
  set_tests_properties(runtime/qpu/multi:fail:prompt PROPERTIES TIMEOUT 180)
  # A QPU that never takes a job: it is given up on, its tasks go to the
  # other one, and those that named it complete as unavailable.
  qpu_addtest(runtime/qpu/multi:down ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -f 0
              ${_qpu_m} --mca device_qpu_devices "mock faults=unavailable:p=1 unavailable_timeout=60|mock"
              --mca device_qpu_retry_backoff 2)
  qpu_addtest(runtime/qpu/multi:down:named ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -f 0 -P half
              ${_qpu_m} --mca device_qpu_devices "mock faults=unavailable:p=1 unavailable_timeout=60|mock"
              --mca device_qpu_retry_backoff 2)
  # Once it is given up on, it is left alone: in a run of two seconds it is
  # asked a dozen times (while its first job is asked again, and once each
  # time it is tried), not the two hundred times of a QPU that is handed
  # every task in turn.
  qpu_addtest(runtime/qpu/multi:down:away ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 16 -l 12 -f 0 -a 40
              ${_qpu_m0} --mca device_qpu_devices "mock faults=unavailable:p=1 unavailable_timeout=50|mock"
              --mca device_qpu_retry_backoff 2 --mca device_qpu_mock_delay const:20000)
  # A QPU that takes no job for the first 300 ms and then does: the tasks
  # go to the other one in the meantime, and it is used again afterwards.
  qpu_addtest(runtime/qpu/multi:back ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 16 -l 12 -u 0
              ${_qpu_m0} --mca device_qpu_devices "mock faults=unavailable:for=300 unavailable_timeout=50|mock"
              --mca device_qpu_retry_backoff 2 --mca device_qpu_mock_delay const:20000)
  # A QPU whose backend cannot be opened gives no device, and the others
  # run the tasks: here the second mock is given a delay it cannot read, and
  # the third one says only when it is opened that it drives a quantum device.
  qpu_addtest(runtime/qpu/multi:noopen ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 24 -l 2 -X 1
              ${_qpu_m} --mca device_qpu_devices "mock|mock delay=nonsense")
  # A QPU that takes its jobs and loses the answer, for the first 400 ms: the
  # submission is given up on, which cancels the job, and its task goes to
  # the other QPU or, later, to this one again. That is another job, under
  # another tag: no backend is asked twice under one tag for two jobs.
  qpu_addtest(runtime/qpu/multi:lost ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 8 -l 2 -u 0
              -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1
              --mca device_qpu_devices "mock faults=lost:for=400 unavailable_timeout=50|mock"
              --mca device_qpu_retry_backoff 2 --mca device_qpu_mock_delay const:300000)
  qpu_addtest(runtime/qpu/multi:lost:noretry ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_multi -n 8 -l 2 -u 0
              -- --mca device_qpu_enabled 1 --mca device_qpu_max_outstanding 1 --mca device_qpu_mock_slots 1
              --mca device_qpu_devices "mock faults=lost:for=400 unavailable_timeout=50|mock"
              --mca device_qpu_retry_backoff 2 --mca device_qpu_mock_delay const:300000 --mca device_qpu_max_retries 0)
  # A list that cannot be read gives no QPU at all, and says so.
  qpu_addtest(runtime/qpu/multi:badlist ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_discovery -x
              -- --mca device_qpu_enabled 1 --mca device_qpu_devices "mock slots=2|mock sluts=2")

  if( MPI_C_FOUND )
    # QPUs on two ranks, one and two of them; and ranks that own none.
    qpu_addtest(runtime/qpu/multi:mp ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_multi -n 24 -l 3 -E 0,1,1
                ${_qpu_m} --mca device_qpu_devices "mock@0|mock@1|mock@1")
    qpu_addtest(runtime/qpu/multi:mp:named ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_multi -n 24 -l 3 -P turn
                ${_qpu_m} --mca device_qpu_devices "mock@0|mock@1|mock@1")
    qpu_addtest(runtime/qpu/multi:mp:4 ${MPI_TEST_CMD_LIST} 4 runtime/qpu/qpu_multi -n 24 -l 3 -P half -E 0,-1,-1
                ${_qpu_m} --mca device_qpu_devices "mock@0|mock@-1|mock@-1")
    qpu_addtest(runtime/qpu/multi:mp:fail ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_multi -n 24 -l 2 -f 1
                ${_qpu_m} --mca device_qpu_devices "mock@0|mock@1 faults=fail:p=1|mock@1" --mca device_qpu_retry_backoff 2)
  endif( MPI_C_FOUND )

  if( MPI_C_FOUND )
    foreach(_np 2 4)
      # The QPU and the quantum tasks on the last rank, their neighbours everywhere.
      qpu_addtest(runtime/qpu/chain:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} runtime/qpu/qpu_chain -n 32 -l 3 -e 5 -q 99
                  ${_qpu} --mca device_qpu_owner_rank -1 --mca device_qpu_mock_delay uniform:200:5000)
      qpu_addtest(runtime/qpu/discovery:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} runtime/qpu/qpu_discovery ${_qpu})
      qpu_addtest(runtime/qpu/termination:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} runtime/qpu/qpu_termination -n 8 -d 100000
                  ${_qpu} --mca device_qpu_mock_delay const:100000)
      # (The dynamic termination detector asserts when a message arrives for a
      # taskpool its rank has not yet marked ready, termdet_fourcounter_module.c
      # in parsec_termdet_fourcounter_incoming_message_start: once in a hundred
      # runs of this test under load, and once in two thousand of a test of
      # that detector without any quantum task. A failure of that kind here is
      # that race, which is the detector's.)
      qpu_addtest(runtime/qpu/termination:dynamic:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} runtime/qpu/qpu_termination_dynamic -n 8 -d 100000
                  ${_qpu} --mca device_qpu_mock_delay const:100000)
      qpu_addtest(runtime/qpu/stress:mp:${_np} ${MPI_TEST_CMD_LIST} ${_np} runtime/qpu/qpu_chain -n 200 -l 5
                  ${_qpu} --mca device_qpu_mock_delay uniform:0:300)
    endforeach()
    qpu_addtest(runtime/qpu/discovery:mp:last ${MPI_TEST_CMD_LIST} 4 runtime/qpu/qpu_discovery -o -1
                ${_qpu} --mca device_qpu_owner_rank -1)
    qpu_addtest(runtime/qpu/chain:cputype:mp ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_chain_cputype -n 32 -l 3 -e 5 -q 1
                ${_qpu} --mca device_qpu_owner_rank 1 --mca device_qpu_mock_delay uniform:200:5000)
    qpu_addtest(runtime/qpu/chain:dtd:mp ${MPI_TEST_CMD_LIST} 2 runtime/qpu/dtd_qpu_chain -n 32 -q 1
                ${_qpu} --mca device_qpu_owner_rank 1 --mca device_qpu_mock_delay const:20000)
    qpu_addtest(runtime/qpu/restart:mp ${MPI_TEST_CMD_LIST} 4 runtime/qpu/qpu_chain -I 10 -n 8 -l 2 -q 1
                ${_qpu} --mca device_qpu_owner_rank 1 --mca device_qpu_mock_delay uniform:0:500)
    # Tasks that share a job, their neighbours on every rank; then with the QPU on the last rank and a job that fails.
    qpu_addtest(runtime/qpu/coalesce:mp ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_coalesce -n 17
                ${_qpu_one} --mca device_qpu_max_batch 4)
    qpu_addtest(runtime/qpu/coalesce:fail:mp ${MPI_TEST_CMD_LIST} 4 runtime/qpu/qpu_coalesce -n 17 -F 8
                ${_qpu_one} --mca device_qpu_max_batch 4 --mca device_qpu_owner_rank -1)
    # The QPU and the quantum tasks on the last rank, the tasks that read their status elsewhere.
    qpu_addtest(runtime/qpu/failure:mp ${MPI_TEST_CMD_LIST} 2 runtime/qpu/qpu_failure -n 40 -m mixed -q 99
                ${_qpu_fail} --mca device_qpu_owner_rank -1)
  endif( MPI_C_FOUND )
endif(PARSEC_HAVE_QPU)
