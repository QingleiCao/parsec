#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Run as a test: cmake -DREPEAT=<n> -P repeat_test.cmake -- <command> [<argument> ...]
#
# Runs a command REPEAT times and fails at the first run that fails, with
# what that run printed. For a check that depends on how threads happen to
# take their turns: one run sees a wrong interleaving only now and then,
# and many runs see it nearly always.

if(NOT REPEAT)
  set(REPEAT 10)
endif()
set(_command)
set(_seen 0)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 0 ${_last})
  if(_seen)
    list(APPEND _command "${CMAKE_ARGV${_i}}")
  elseif("${CMAKE_ARGV${_i}}" STREQUAL "--")
    set(_seen 1)
  endif()
endforeach()
if(NOT _command)
  message(FATAL_ERROR "no command after --")
endif()

foreach(_run RANGE 1 ${REPEAT})
  execute_process(COMMAND ${_command} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "run ${_run} of ${REPEAT} failed (${_rc}):\n${_out}\n${_err}")
  endif()
endforeach()
message(STATUS "${REPEAT} runs passed")
