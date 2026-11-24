#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Runtime tests of the QPU layer. They all use the mock backend: none of them
# can reach a quantum device (see check_no_hardware.cmake).

if(PARSEC_HAVE_QPU)
  include(ParsecCompilePTG)

  add_library(qpu_test_common OBJECT qpu_test_common.c)
  target_link_libraries(qpu_test_common PRIVATE parsec)

  # An executable of this directory: it finds the local headers and links
  # what the QPU tests share.
  function(qpu_addtest_executable target)
    parsec_addtest_executable(C ${target} ${ARGN})
    target_include_directories(${target} PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(${target} PRIVATE qpu_test_common tests_runtime_common)
  endfunction(qpu_addtest_executable)

  qpu_addtest_executable(qpu_chain)
  target_ptg_sources(qpu_chain PRIVATE "qpu_chain.jdf")

  # The same test with a quantum task whose body is not of the QPU type: such
  # a body may submit as well, and the job goes to the QPU of the process all
  # the same. The type is one word of the JDF, so the source is copied with
  # that word taken out.
  file(READ qpu_chain.jdf _qpu_chain_jdf)
  string(REPLACE "BODY [type=QPU]" "BODY" _qpu_chain_jdf "${_qpu_chain_jdf}")
  string(REPLACE "parsec_qpu_chain_" "parsec_qpu_chain_cputype_" _qpu_chain_jdf "${_qpu_chain_jdf}")
  string(REPLACE "PARSEC_qpu_chain_" "PARSEC_qpu_chain_cputype_" _qpu_chain_jdf "${_qpu_chain_jdf}")
  file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qpu_chain_cputype.jdf
       CONTENT "${_qpu_chain_jdf}" @ONLY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS qpu_chain.jdf)
  qpu_addtest_executable(qpu_chain_cputype)
  target_ptg_sources(qpu_chain_cputype PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/qpu_chain_cputype.jdf")

  qpu_addtest_executable(qpu_discovery)
  target_ptg_sources(qpu_discovery PRIVATE "qpu_discovery.jdf")

  # CPU, CUDA and QPU bodies in one task graph. It builds without CUDA too:
  # the CUDA bodies are then left out and the CPU bodies of the same task
  # classes run.
  qpu_addtest_executable(cpu_gpu_qpu)
  target_ptg_sources(cpu_gpu_qpu PRIVATE "cpu_gpu_qpu.jdf")

  qpu_addtest_executable(qpu_overlap)
  target_ptg_sources(qpu_overlap PRIVATE "qpu_overlap.jdf")

  qpu_addtest_executable(qpu_termination)
  target_ptg_sources(qpu_termination PRIVATE "qpu_termination.jdf")

  # The same test with the four-counter termination detector. The detector is
  # chosen by a JDF option, so the source is copied with that one line changed.
  # The detector exists only in an MPI build: without MPI a taskpool that asks
  # for it crashes when it is created.
  if(MPI_C_FOUND)
    file(READ qpu_termination.jdf _qpu_termination_jdf)
    string(REPLACE "/* termdet: local */" "%option termdet = \"dynamic\""
           _qpu_termination_jdf "${_qpu_termination_jdf}")
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qpu_termination_dynamic.jdf
         CONTENT "${_qpu_termination_jdf}" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS qpu_termination.jdf)
    qpu_addtest_executable(qpu_termination_dynamic)
    target_ptg_sources(qpu_termination_dynamic PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/qpu_termination_dynamic.jdf")
    target_compile_definitions(qpu_termination_dynamic PRIVATE QPU_TERMINATION_DYNAMIC)
  endif(MPI_C_FOUND)

  qpu_addtest_executable(dtd_qpu_chain SOURCES dtd_qpu_chain.c)

  # The sampler and the mock backend on their own: no runtime is started.
  qpu_addtest_executable(qpu_sampler_test SOURCES qpu_sampler_test.c)
  target_link_libraries(qpu_sampler_test PRIVATE m)
  qpu_addtest_executable(qpu_mock_test SOURCES qpu_mock_test.c)
  target_link_libraries(qpu_mock_test PRIVATE m)

  qpu_addtest_executable(qpu_failure)
  target_ptg_sources(qpu_failure PRIVATE "qpu_failure.jdf")

  qpu_addtest_executable(qpu_coalesce)
  target_ptg_sources(qpu_coalesce PRIVATE "qpu_coalesce.jdf")

  # Experiments E1 and E2: what a quantum task costs, with and without a
  # worker held during its job.
  qpu_addtest_executable(qpu_bench)
  target_ptg_sources(qpu_bench PRIVATE "qpu_bench.jdf")

  qpu_addtest_executable(qpu_guard)
  target_ptg_sources(qpu_guard PRIVATE "qpu_guard.jdf")

  qpu_addtest_executable(qpu_misuse)
  target_ptg_sources(qpu_misuse PRIVATE "qpu_misuse.jdf")
endif(PARSEC_HAVE_QPU)
