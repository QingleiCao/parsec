#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Run as a test: cmake -DTESTS_DIR=<tests directory> -P check_no_hardware.cmake
#
# No registered test may allow a QPU backend that drives a physical device.
# Such a backend opens only when device_qpu_allow_hardware is set, so it is
# enough that no test definition sets it to anything but 0.

if(NOT TESTS_DIR)
  message(FATAL_ERROR "TESTS_DIR is not set")
endif()

file(GLOB_RECURSE _test_definitions "${TESTS_DIR}/Testings.cmake" "${TESTS_DIR}/*/Testings.cmake")
list(REMOVE_DUPLICATES _test_definitions)
set(_offenders)
foreach(_file ${_test_definitions})
  file(STRINGS "${_file}" _lines REGEX "allow_hardware")
  foreach(_line ${_lines})
    # device_qpu_allow_hardware followed, after a space or an equal sign, by a non-zero value
    if(_line MATCHES "allow_hardware[ =\"]+[1-9]")
      list(APPEND _offenders "${_file}: ${_line}")
    endif()
  endforeach()
endforeach()

list(LENGTH _test_definitions _nb)
if(_offenders)
  string(REPLACE ";" "\n  " _offenders "${_offenders}")
  message(FATAL_ERROR "These test definitions allow a physical QPU backend:\n  ${_offenders}")
endif()
message(STATUS "Checked ${_nb} test definition files: none allows a physical QPU backend")
