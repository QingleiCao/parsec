#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(ParsecCompilePTG)

parsec_addtest_executable(C schedmicro SOURCES main.c ep_wrapper.c schedmicro_data.c)
target_ptg_sources(schedmicro PRIVATE "ep.jdf")
target_link_libraries(schedmicro PRIVATE m tests_runtime_common)

parsec_addtest_executable(C executed_tasks SOURCES executed_tasks.c ep_wrapper.c schedmicro_data.c)
target_ptg_sources(executed_tasks PRIVATE "ep.jdf")
target_link_libraries(executed_tasks PRIVATE tests_runtime_common)
