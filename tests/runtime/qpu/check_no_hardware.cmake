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
#
# A second lock on the same door: a test definition that lists its QPUs
# (device_qpu_devices) may name only the mock and the backends the guard test
# registers itself, which drive nothing.

if(NOT TESTS_DIR)
  message(FATAL_ERROR "TESTS_DIR is not set")
endif()

file(GLOB_RECURSE _test_definitions "${TESTS_DIR}/Testings.cmake" "${TESTS_DIR}/*/Testings.cmake")
list(REMOVE_DUPLICATES _test_definitions)
set(_offenders)
set(_nb_lists 0)
foreach(_file ${_test_definitions})
  file(STRINGS "${_file}" _lines REGEX "allow_hardware")
  foreach(_line ${_lines})
    # device_qpu_allow_hardware followed, after a space or an equal sign, by a non-zero value
    if(_line MATCHES "allow_hardware[ =\"]+[1-9]")
      list(APPEND _offenders "${_file}: ${_line}")
    endif()
  endforeach()

  file(STRINGS "${_file}" _lines REGEX "device_qpu_devices")
  foreach(_line ${_lines})
    if(_line MATCHES "^[ \t]*#")
      continue()
    endif()
    if(_line MATCHES "device_qpu_devices[ ]+\"([^\"]*)\"")
      set(_list "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "device_qpu_devices[ ]+([^ )\"]+)")
      set(_list "${CMAKE_MATCH_1}")
    else()
      # The list is not on the line that names it: not something this script can read.
      list(APPEND _offenders "${_file}: ${_line} (the list of QPUs has to follow device_qpu_devices on the same line)")
      continue()
    endif()
    math(EXPR _nb_lists "${_nb_lists} + 1")
    string(REPLACE "|" ";" _entries "${_list}")
    foreach(_entry ${_entries})
      string(STRIP "${_entry}" _entry)
      string(REGEX REPLACE "[@ ].*$" "" _backend "${_entry}")
      if(NOT _backend MATCHES "^(mock|guard_physical|guard_late|guard_missing)$")
        list(APPEND _offenders "${_file}: ${_line} (backend '${_backend}')")
      endif()
    endforeach()
  endforeach()
endforeach()

list(LENGTH _test_definitions _nb)
if(_offenders)
  string(REPLACE ";" "\n  " _offenders "${_offenders}")
  message(FATAL_ERROR "These test definitions allow a physical QPU backend:\n  ${_offenders}")
endif()
message(STATUS "Checked ${_nb} test definition files: none allows a physical QPU backend, "
               "and the ${_nb_lists} lists of QPUs name no backend but the mock and the guard test's own")
