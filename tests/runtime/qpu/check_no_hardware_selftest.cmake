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
# pass either. A case may name its rule, by words of what the script says
# when that rule refuses: then it has to be refused by that rule, and by that
# rule alone.

if(NOT SCRIPT OR NOT WORK)
  message(FATAL_ERROR "SCRIPT and WORK have to be set")
endif()
file(REMOVE_RECURSE "${WORK}")

set(_names)
function(a_definition name verdict text)
  file(WRITE "${WORK}/${name}/Testings.cmake" "${text}\n")
  set(_names ${_names} "${name}:${verdict}" PARENT_SCOPE)
  set(_rule_${name} "${ARGN}" PARENT_SCOPE)
endfunction()

# The rules hold: the three adapters a test may start, over several lines, with a comment in between;
# and a --target that is cmake's, not an adapter's.
a_definition(good accepted [=[
  qpu_addtest(a ${SHM_TEST_CMD_LIST} runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock
              runtime/qpu/qpu_adapter_loopback -s runtime/qpu/a.sock -P -N
              # --backend ibm_real in a comment is no backend
              -- runtime/qpu/qpu_chain -- --mca device_qpu_allow_hardware 0
              --mca device_qpu_devices "mock|adapter address=runtime/qpu/a.sock")
  set(_qpu_aer runtime/qpu/qpu_adapter_run -w 180 -s runtime/qpu/b.sock
      "${_qpu_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_qiskit.py --socket runtime/qpu/b.sock --backend aer)
  qpu_addtest(b ${_qpu_aer} --device generic:5:line --no-noise -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/b.sock)
  set(_qpu_cudaq runtime/qpu/qpu_adapter_run -w 180 -s runtime/qpu/c.sock
      "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py --socket runtime/qpu/c.sock --target qpp-cpu)
  qpu_addtest(c ${_qpu_cudaq} -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/c.sock)
  qpu_addtest(d runtime/qpu/qpu_adapter_run -s runtime/qpu/d.sock
              "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py --socket runtime/qpu/d.sock
              --target nvidia --option mqpu --qpu 1 -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/d.sock)
  qpu_addtest(e "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py --check-reader runtime/qpu/qasm_texts)
  add_test(NAME f COMMAND ${CMAKE_COMMAND} --build . --target must_fail_something)
  add_test(NAME g COMMAND ${CMAKE_COMMAND} --build . --config Release --target must_fail_something_else)
  qpu_addtest(h ${MPIEXEC_EXECUTABLE} --machinefile hosts -n 2 runtime/qpu/qpu_chain -- --mca device_qpu_enabled 1)
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
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock /opt/vendor/bin/vendor_gateway --endpoint ibm_real
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=] "neither the adapter of the tests")
# The launcher written so that its adapter is not where the script looks for it.
a_definition(launcher_with_a_time_that_is_no_integer refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -w 60.5 -s runtime/qpu/a.sock /opt/vendor/bin/vendor_gateway
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=] "neither the adapter of the tests")
a_definition(launcher_with_a_quoted_option refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run "-s" runtime/qpu/a.sock /opt/vendor/bin/vendor_gateway
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=] "a launcher this script cannot read")
a_definition(launcher_in_a_variable refused [=[
  set(_run runtime/qpu/qpu_adapter_run)
  qpu_addtest(a ${_run} -s runtime/qpu/a.sock /opt/vendor/bin/vendor_gateway
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=] "a launcher this script cannot read")
a_definition(a_bracket_in_a_socket refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a[1.sock /opt/vendor/bin/vendor_gateway
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
  qpu_addtest(b runtime/qpu/qpu_adapter_run -s runtime/qpu/b.sock runtime/qpu/qpu_adapter_loopback -s runtime/qpu/b.sock
              -- runtime/qpu/qpu_chain -- --mca device_qpu_backend adapter)
]=] "neither the adapter of the tests")
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
# The adapter for CUDA-Q.
a_definition(cudaq_another_target refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --target quantinuum -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_with_a_longer_name refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --target nvidia-remote -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_with_an_equal_sign refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --target=ionq -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_on_the_next_line refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --target
              iqm -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_in_a_variable refused [=[
  set(_qpu_cudaq runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock
      "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py --socket runtime/qpu/a.sock)
  qpu_addtest(a ${_qpu_cudaq} --target braket -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_another_option refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --target nvidia --option remote-mqpu -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "option of its target other than mqpu")
a_definition(cudaq_a_qpu_by_its_name refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --qpu helios -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "a QPU that is named")
a_definition(cudaq_a_machine refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --machine H2-1 -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "belongs to an account")
a_definition(cudaq_adapter_of_someone_else refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" /home/someone/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=])
a_definition(cudaq_with_the_other_python refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock /usr/bin/env ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "neither the adapter of the tests")
a_definition(cudaq_target_quoted refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock "--target" ionq -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_in_brackets refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock [[--target]] ionq -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "target that is not a simulator")
a_definition(cudaq_target_behind_what_is_not_a_build refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a--build "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a--build --verbose --target ionq -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a--build)
]=] "target that is not a simulator")
a_definition(cudaq_target_without_its_value refused [=[
  set(_t --target)
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock ${_t} ionq -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "not followed by its value")
a_definition(cudaq_credentials refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock --credentials /home/someone/key -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
]=] "belongs to an account")
a_definition(cudaq_told_what_to_load refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock  -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
  set_tests_properties(a PROPERTIES ENVIRONMENT "CUDAQ_DEFAULT_SIMULATOR=remote-mqpu")
]=] "decides what an adapter loads")
a_definition(cudaq_with_a_path_for_python refused [=[
  qpu_addtest(a runtime/qpu/qpu_adapter_run -s runtime/qpu/a.sock "${_qpu_cudaq_python}" ${PROJECT_SOURCE_DIR}/tools/qpu/parsec_qpu_adapter_cudaq.py
              --socket runtime/qpu/a.sock  -- runtime/qpu/qpu_adapter_stat -a runtime/qpu/a.sock)
  set_tests_properties(a PROPERTIES ENVIRONMENT "PYTHONPATH=/home/someone/lib")
]=] "decides what an adapter loads")
a_definition(a_token_of_another_vendor refused [=[
  set_tests_properties(a PROPERTIES ENVIRONMENT "OQC_AUTH_TOKEN=abc")
]=] "belongs to an account")
a_definition(a_secret refused [=[
  set_tests_properties(a PROPERTIES ENVIRONMENT "AWS_SECRET_ACCESS_KEY=abc")
]=] "belongs to an account")
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
  elseif(_verdict STREQUAL "refused" AND _rule_${_name})
    # The rule this case is a case of, and no other: one line of the script's, with the rule's words.
    string(FIND "${_err}" "${_rule_${_name}}" _at)
    string(REGEX MATCHALL "Testings[.]cmake: " _lines "${_err}")
    list(LENGTH _lines _nb_lines)
    if(_at EQUAL -1 OR NOT _nb_lines EQUAL 1)
      list(APPEND _wrong "${_name}: to be refused by one rule, '${_rule_${_name}}', and it was refused so:\n${_err}")
    endif()
  endif()
endforeach()
file(REMOVE_RECURSE "${WORK}")
list(LENGTH _names _nb)
if(_wrong)
  string(REPLACE ";" "\n" _wrong "${_wrong}")
  message(FATAL_ERROR "check_no_hardware.cmake does not keep its rules:\n${_wrong}")
endif()
message(STATUS "check_no_hardware.cmake accepted the definition that keeps its rules and refused the others (${_nb} definitions)")
