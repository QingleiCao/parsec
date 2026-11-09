#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#

parsec_addtest_cmd(dsl/ptg/choice ${SHM_TEST_CMD_LIST} dsl/ptg/choice/choice 8)
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/choice:mp ${MPI_TEST_CMD_LIST} 4 dsl/ptg/choice/choice 8)
endif( MPI_C_FOUND )
