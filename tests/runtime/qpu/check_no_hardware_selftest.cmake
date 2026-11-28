#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Run as a test: cmake -DSCRIPT=<check_no_hardware.cmake> -DWORK=<a directory to write in> -P check_no_hardware_selftest.cmake
#
# check_no_hardware.cmake keeps the test definitions away from quantum
# devices. This gives it test definitions that break its rules, one rule
# each, and it has to refuse every one of them; and one that breaks none,
# which it has to accept, so that a script that refuses everything does not
# pass either.

if(NOT SCRIPT OR NOT WORK)
  message(FATAL_ERROR "SCRIPT and WORK have to be set")
endif()
file(REMOVE_RECURSE "${WORK}")

set(_names)
function(a_definition name verdict text)
  file(WRITE "${WORK}/${name}/Testings.cmake" "${text}\n")
  set(_names ${_names} "${name}:${verdict}" PARENT_SCOPE)
endfunction()

# The rules hold: the two adapters a test may start, over several lines, with a comment in between.
a_definition(good accepted [=[
  qpu_addtest(a ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock
              runtime/qpu/qpu_adapter_loopback -s runtime/qpu/a.sock -P -N
              # --backend ibm_real in a comment is no backend
              -- runtime/qpu/qpu_chain -- --mca device_qpu_allow_hardware 0
              --mca device_qpu_devices "mock|adapter address=runtime/qpu/a.sock")
  set(_qpu_aer runtime/qpu/qpu_adapter_run -w 180 -s runtime/qpu/b.sock
      "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py --socket runtime/qpu/b.sock --backend aer)
  qpu_addtest(b ${_qpu_aer} --device generic:5:line --no-noise -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/b.sock)
]=])
# The first lock.
a_definition(allow_hardware refused [=[
  qpu_addtest(a runtime/qpu/qpu_chain -- --mca device_qpu_enabled 1 --mca device_qpu_allow_hardware 1)
]=])
# The second lock.
a_definition(backend_in_a_list refused [=[
  qpu_addtest(a runtime/qpu/qpu_chain -- --mca device_qpu_devices "mock|qrmi resource=somewhere")
]=])
a_definition(list_on_the_next_line refused [=[
  qpu_addtest(a runtime/qpu/qpu_chain -- --mca device_qpu_devices
              "mock|mock")
]=])
# The third lock.
a_definition(another_program refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock /opt/vendor/bin/vendor_gateway --qpu ibm_real
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=])
a_definition(another_adapter_script refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" /home/someone/my_adapter_ibm.py --socket runtime/qpu/a.sock
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=])
a_definition(an_adapter_started_another_way refused [=[
  qpu_addtest(a sh -c "/home/someone/my_adapter_ibm.py --socket runtime/qpu/a.sock & runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter")
]=])
a_definition(backend_with_a_longer_name refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py
              --socket runtime/qpu/a.sock --backend aer2 -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(backend_on_the_next_line refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py
              --socket runtime/qpu/a.sock --backend
              ibm_real -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(backend_with_an_equal_sign refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py
              --socket runtime/qpu/a.sock --backend=ibm -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(a_real_device refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py
              --socket runtime/qpu/a.sock --backend aer --device ibm_brisbane -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(allow_hardware_of_the_adapter refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py
              --socket runtime/qpu/a.sock --backend aer --allow-hardware -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(a_token refused [=[
  set_tests_properties(a PROPERTIES ENVIRONMENT "QISKIT_IBM_TOKEN=abc")
]=])

set(_wrong)
foreach(_entry ${_names})
  string(REGEX REPLACE ":.*$" "" _name "${_entry}")
  string(REGEX REPLACE "^.*:" "" _verdict "${_entry}")
  execute_process(COMMAND ${CMAKE_COMMAND} -DTESTS_DIR=${WORK}/${_name} -P ${SCRIPT}
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if((_verdict STREQUAL "accepted" AND NOT _rc EQUAL 0) OR (_verdict STREQUAL "refused" AND _rc EQUAL 0))
    list(APPEND _wrong "${_name}: expected to be ${_verdict} (${_rc})\n${_err}")
  elseif(_verdict STREQUAL "refused" AND NOT "${_err}" MATCHES "These test definitions allow a physical QPU backend")
    list(APPEND _wrong "${_name}: refused, but not by a rule of the script:\n${_err}")
  endif()
endforeach()
file(REMOVE_RECURSE "${WORK}")
list(LENGTH _names _nb)
if(_wrong)
  string(REPLACE ";" "\n" _wrong "${_wrong}")
  message(FATAL_ERROR "check_no_hardware.cmake does not keep its rules:\n${_wrong}")
endif()
message(STATUS "check_no_hardware.cmake accepted the definition that keeps its rules and refused the others (${_nb} definitions)")
