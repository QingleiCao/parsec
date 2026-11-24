#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Block-decomposed QAOA for weighted MaxCut. The arithmetic and the
# sequential reference need the QPU component for its circuit format and its
# sampler, and nothing else; testing_qaoa is the application on the runtime.

if(PARSEC_HAVE_QPU)
  # Graphs, the steps of the algorithm, the exact solvers and the sequential
  # reference: shared by every program of this directory.
  add_library(qaoa_common OBJECT qaoa_graph.c qaoa_core.c qaoa_exact.c qaoa_seq.c)
  target_link_libraries(qaoa_common PRIVATE parsec)

  function(qaoa_addtest_executable target)
    parsec_addtest_executable(C ${target} ${ARGN})
    target_include_directories(${target} PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(${target} PRIVATE qaoa_common m)
  endfunction(qaoa_addtest_executable)

  qaoa_addtest_executable(qaoa_math_test SOURCES qaoa_math_test.c)
  qaoa_addtest_executable(qaoa_reference SOURCES qaoa_reference.c)

  # The application: one PTG taskpool whose quantum tasks have a body of the
  # QPU type, its data, and a driver that compares each run with the
  # sequential reference.
  include(ParsecCompilePTG)
  qaoa_addtest_executable(testing_qaoa SOURCES testing_qaoa.c qaoa_data.c)
  target_ptg_sources(testing_qaoa PRIVATE "qaoa.jdf")
  target_link_libraries(testing_qaoa PRIVATE tests_runtime_common)
endif(PARSEC_HAVE_QPU)
