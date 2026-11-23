#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

# Without MPI the launcher is empty, and the process count below would then be
# taken for the name of the executable.
if( MPI_C_FOUND )
  set(_schedmicro_launcher ${MPI_TEST_CMD_LIST} 1)
else( MPI_C_FOUND )
  set(_schedmicro_launcher ${SHM_TEST_CMD_LIST})
endif( MPI_C_FOUND )

foreach(_sched ${MCA_sched})
    parsec_addtest_cmd(runtime/scheduling:${_sched} ${_schedmicro_launcher} runtime/scheduling/schedmicro -t 10 -l 8 -n 512 -- --mca mca_sched ${_sched})
endforeach()

# A thread that waits inside a task must not keep other tasks to itself. Only
# for lfq, which lets a thread take several tasks at once from the queue that
# the threads share. The test is not meant for every scheduler: ip executes the
# tasks that ask to be executed again before all the others, and never gets to
# the rest.
if( "lfq" IN_LIST MCA_sched )
    parsec_addtest_cmd(runtime/scheduling/stalled:lfq ${_schedmicro_launcher} runtime/scheduling/stalled -c 4 -- --mca mca_sched lfq)
endif( "lfq" IN_LIST MCA_sched )

if( MPI_C_FOUND )
  foreach(_sched ${MCA_sched})
        parsec_addtest_cmd(runtime/scheduling:mp:${_sched} ${MPI_TEST_CMD_LIST} 2 runtime/scheduling/schedmicro -t 10 -l 8 -n 512 -- --mca mca_sched ${_sched})
    endforeach()
endif( MPI_C_FOUND )
