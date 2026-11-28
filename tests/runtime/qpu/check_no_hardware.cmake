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
# three this script knows: the loopback of the tests and the two adapters of
# tools/qpu. The one for Qiskit with no backend but its simulator and no
# device but the made-up model; the one for CUDA-Q with no target but its two
# simulators; and neither with an option that names a device or an account.
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
  # A bracket is a blank here: CMake would not take the matches below one by
  # one with a bracket in one of them, and an option in brackets is an option.
  string(REGEX REPLACE "[][]" " " _text "${_text}")
  # The options are looked for in the text without its quotes: "--target" is --target.
  string(REPLACE "\"" "" _plain "${_text}")

  # What runs the jobs of an adapter program: the simulator, by its whole name.
  string(REGEX MATCHALL "--backend[ =]+[^ )]+" _found "${_plain}")
  foreach(_one ${_found})
    string(REGEX REPLACE "^--backend[ =]+" "" _value "${_one}")
    if(NOT _value STREQUAL "aer")
      list(APPEND _offenders "${_file}: ${_one} (an adapter with a backend other than the simulator)")
    endif()
  endforeach()
  # The device it simulates: none, or the made-up model.
  string(REGEX MATCHALL "--device[ =]+[^ )]+" _found "${_plain}")
  foreach(_one ${_found})
    string(REGEX REPLACE "^--device[ =]+" "" _value "${_one}")
    if(NOT _value MATCHES "^(none|generic:[0-9]+(:(line|ring|full))?)$")
      list(APPEND _offenders "${_file}: ${_one} (an adapter with a device that is not the made-up model)")
    endif()
  endforeach()
  # In a file that names the adapter for CUDA-Q, and in no other, so that the
  # options of other programs are theirs:
  if(_plain MATCHES "parsec_qpu_adapter_cudaq")
    # its target is the simulator of CUDA-Q on the CPU or the one on a GPU, by
    # its whole name. (cmake --build has a --target too: those are taken out
    # first, as cmake is given them.)
    string(REGEX REPLACE "([$][{]CMAKE_COMMAND[}] --build [^ )-][^ )]*)( --(config|parallel|preset) [^ )]+)*" "\\1" _no_builds "${_plain}")
    string(REGEX REPLACE "[$][{]CMAKE_COMMAND[}] --build [^ )-][^ )]* --target [^ )]+" "" _no_builds "${_no_builds}")
    string(REGEX MATCHALL "--target[ =]+[^ )]+" _found "${_no_builds}")
    foreach(_one ${_found})
      string(REGEX REPLACE "^--target[ =]+" "" _value "${_one}")
      if(NOT _value MATCHES "^(qpp-cpu|nvidia)$")
        list(APPEND _offenders "${_file}: ${_one} (an adapter with a target that is not a simulator of CUDA-Q)")
      endif()
    endforeach()
    # The option of that target: a QPU per GPU, and nothing else.
    string(REGEX MATCHALL "--option[ =]+[^ )]+" _found "${_plain}")
    foreach(_one ${_found})
      string(REGEX REPLACE "^--option[ =]+" "" _value "${_one}")
      if(NOT _value STREQUAL "mqpu")
        list(APPEND _offenders "${_file}: ${_one} (an adapter with an option of its target other than mqpu)")
      endif()
    endforeach()
    # Which QPU of the target: a number, never a name.
    string(REGEX MATCHALL "--qpu[ =]+[^ )]+" _found "${_plain}")
    foreach(_one ${_found})
      string(REGEX REPLACE "^--qpu[ =]+" "" _value "${_one}")
      if(NOT _value MATCHES "^[0-9]+$")
        list(APPEND _offenders "${_file}: ${_one} (a QPU that is named, where an adapter takes a number)")
      endif()
    endforeach()
    # One of these options with its value somewhere else (in a variable, in
    # another command) is one this script cannot read.
    if(_no_builds MATCHES "(--target|--option|--qpu)[ =]*\\)")
      list(APPEND _offenders "${_file}: '${CMAKE_MATCH_1}' (an option of an adapter that is not followed by its value)")
    endif()
  endif()
  if(_plain MATCHES "(--backend|--device)[ =]*\\)")
    list(APPEND _offenders "${_file}: '${CMAKE_MATCH_1}' (an option of an adapter that is not followed by its value)")
  endif()
  # Nothing that names an account, or that an adapter would need to leave the simulator.
  if(_plain MATCHES "(--allow-hardware|--token|--instance|--channel|--account|--url|--machine[ =]|--credentials|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]|IBM_TOKEN|QISKIT_IBM|[A-Z_]*TOKEN[A-Z_]*=|[A-Z_]*CREDENTIALS[A-Z_]*=|[A-Z_]*SECRET[A-Z_]*=|ACCESS_KEY)")
    list(APPEND _offenders "${_file}: '${CMAKE_MATCH_1}' (an option or a name that belongs to an account or a device)")
  endif()
  # In a file that names an adapter program, nothing in the environment that
  # tells it or its libraries what to load or what to run on.
  if(_plain MATCHES "parsec_qpu_adapter_" AND _plain MATCHES "(CUDAQ_[A-Z_]+=|IQM_[A-Z_]+=|OQC_[A-Z_]+=|IONQ_[A-Z_]+=|PYTHONPATH=)")
    list(APPEND _offenders "${_file}: '${CMAKE_MATCH_1}' (a variable of the environment that decides what an adapter loads or runs on)")
  endif()
  # The program that is started beside a test: the adapter of the tests, or
  # one of the two adapters of tools/qpu, each with its own Python, and no other.
  string(REGEX MATCHALL "qpu_adapter_run( -w [0-9.]+)? -s [^ ]+ [^ ]+( [^ ]+)?" _found "${_text}")
  foreach(_one ${_found})
    math(EXPR _nb_runs "${_nb_runs} + 1")
    if(NOT _one MATCHES " -s [^ ]+ runtime/qpu/qpu_adapter_loopback( |$)" AND
       NOT _one MATCHES " -s [^ ]+ \"[$][{]_qpu_python[}]\" [$][{]PROJECT_SOURCE_DIR[}]/tools/qpu/parsec_qpu_adapter_qiskit[.]py$" AND
       NOT _one MATCHES " -s [^ ]+ \"[$][{]_qpu_cudaq_python[}]\" [$][{]PROJECT_SOURCE_DIR[}]/tools/qpu/parsec_qpu_adapter_cudaq[.]py$")
      list(APPEND _offenders "${_file}: ${_one} (a test starts an adapter that is neither the adapter of the tests nor one of tools/qpu)")
    endif()
  endforeach()
  # ... and every qpu_adapter_run of the file is one that was read just now:
  # written another way (its options quoted, a time that is no number, the
  # program in a variable), it would start what this script did not see.
  string(REGEX MATCHALL "qpu_adapter_run" _named "${_text}")
  list(LENGTH _found _nb_found)
  list(LENGTH _named _nb_named)
  if(NOT _nb_found EQUAL _nb_named)
    list(APPEND _offenders "${_file}: qpu_adapter_run is named ${_nb_named} times and followed ${_nb_found} times by [-w seconds] -s socket and an adapter (a launcher this script cannot read)")
  endif()
  # ... and no other adapter program is named at all.
  string(REGEX MATCHALL "[^ \"()]*adapter[^ \"()]*[.]py" _found "${_text}")
  foreach(_one ${_found})
    if(NOT _one STREQUAL "\${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py" AND
       NOT _one STREQUAL "\${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py")
      list(APPEND _offenders "${_file}: ${_one} (an adapter program other than the two of tools/qpu)")
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
               "and the ${_nb_runs} adapters that tests start are the adapter of the tests or a simulator of tools/qpu")
