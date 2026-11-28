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
# (device_qpu_devices) may name only the mock, the backends the guard test
# registers itself, which drive nothing, and the adapter backend. That one is
# the client of a separate program, and what that program drives is not in
# the test definition: so a third lock. A test may start no adapter but the
# two this script knows, the loopback of the tests and the adapter of
# tools/qpu, and that one with no backend but its simulator, no device but the
# made-up model, and no option that names a device or an account.
# check_no_hardware_selftest.cmake gives this script definitions that break
# each rule, and it has to refuse every one.

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
      if(NOT _backend MATCHES "^(mock|adapter|guard_physical|guard_late|guard_missing)$")
        list(APPEND _offenders "${_file}: ${_line} (backend '${_backend}')")
      endif()
    endforeach()
  endforeach()
endforeach()

# The third lock: the adapters tests start. Each definition file is read as
# one text, its comment lines taken out and its blanks and line ends made one
# blank each, so that a command written over several lines is one command.
set(_nb_runs 0)
foreach(_file ${_test_definitions})
  file(READ "${_file}" _text)
  string(REGEX REPLACE "(^|\n)[ \t]*#[^\n]*" "\\1" _text "${_text}")
  string(REGEX REPLACE "[ \t\r\n]+" " " _text "${_text}")
  string(REPLACE ";" " " _text "${_text}")

  # What runs the jobs of an adapter program: the simulator, by its whole name.
  string(REGEX MATCHALL "--backend[ =]+[^ )]+" _found "${_text}")
  foreach(_one ${_found})
    string(REGEX REPLACE "^--backend[ =]+" "" _value "${_one}")
    string(REPLACE "\"" "" _value "${_value}")
    if(NOT _value STREQUAL "aer")
      list(APPEND _offenders "${_file}: ${_one} (an adapter with a backend other than the simulator)")
    endif()
  endforeach()
  # The device it simulates: none, or the made-up model.
  string(REGEX MATCHALL "--device[ =]+[^ )]+" _found "${_text}")
  foreach(_one ${_found})
    string(REGEX REPLACE "^--device[ =]+" "" _value "${_one}")
    string(REPLACE "\"" "" _value "${_value}")
    if(NOT _value MATCHES "^(none|generic:[0-9]+(:(line|ring|full))?)$")
      list(APPEND _offenders "${_file}: ${_one} (an adapter with a device that is not the made-up model)")
    endif()
  endforeach()
  # Nothing that names an account, or that an adapter would need to leave the simulator.
  if(_text MATCHES "(--allow-hardware|--token|--instance|--channel|--account|--url|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]|IBM_TOKEN|QISKIT_IBM)")
    list(APPEND _offenders "${_file}: '${CMAKE_MATCH_1}' (an option or a name that belongs to an account or a device)")
  endif()
  # The program that is started beside a test: the adapter of the tests, or
  # the adapter of tools/qpu, and no other.
  string(REGEX MATCHALL "qpu_adapter_run( -w [0-9]+)? -s [^ ]+ [^ ]+( [^ ]+)?" _found "${_text}")
  foreach(_one ${_found})
    math(EXPR _nb_runs "${_nb_runs} + 1")
    if(NOT _one MATCHES " -s [^ ]+ runtime/qpu/qpu_adapter_loopback( |$)" AND
       NOT _one MATCHES " -s [^ ]+ \"[$][{]_qpu_python[}]\" [$][{]PROJECT_SOURCE_DIR[}]/tools/qpu/parsec_qpu_adapter_qiskit[.]py$")
      list(APPEND _offenders "${_file}: ${_one} (a test starts an adapter that is neither the adapter of the tests nor the one of tools/qpu)")
    endif()
  endforeach()
  # ... and no other adapter program is named at all.
  string(REGEX MATCHALL "[^ \"()]*adapter[^ \"()]*[.]py" _found "${_text}")
  foreach(_one ${_found})
    if(NOT _one STREQUAL "\${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py")
      list(APPEND _offenders "${_file}: ${_one} (an adapter program other than the one of tools/qpu)")
    endif()
  endforeach()
endforeach()

list(LENGTH _test_definitions _nb)
if(_offenders)
  string(REPLACE ";" "\n  " _offenders "${_offenders}")
  message(FATAL_ERROR "These test definitions allow a physical QPU backend:\n  ${_offenders}")
endif()
message(STATUS "Checked ${_nb} test definition files: none allows a physical QPU backend, "
               "the ${_nb_lists} lists of QPUs name no backend but the mock, the adapter and the guard test's own, "
               "and the ${_nb_runs} adapters that tests start are the adapter of the tests or the simulator of tools/qpu")
