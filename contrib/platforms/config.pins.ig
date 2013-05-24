#!/bin/bash                                                                       

echo "### Defaults for ig machine"
echo "# Many parameters can be tuned (command line, env, etc)"
echo "# Open this file to see what common variables are available"
echo "#"

HOSTNAME=`/bin/uname -n`

# This option permits setting arbitrary options to cmake
# Options passed on the command line are appended to this variable
USER_OPTIONS+=" -DDPLASMA_SCHED_HWLOC=ON"
USER_OPTIONS+=" -DDAGUE_DIST_WITH_MPI=OFF"
USER_OPTIONS+=" -DDAGUE_GPU_WITH_CUDA=OFF"
USER_OPTIONS+=" -DPYTHON_LIBRARIES:FILEPATH=$HOME/enthought_7.3.2_x64/lib/libpython2.7.so"
USER_OPTIONS+=" -DPYTHON_INCLUDE_DIRS:FILEPATH=$HOME/enthought_7.3.2_x64/include"
USER_OPTIONS+=" -DPYTHON_EXECUTABLE:FILEPATH=$HOME/enthought_7.3.2_x64/bin/python"
USER_OPTIONS+=" -DPAPI_DIR=$HOME/sw/$HOSTNAME"
USER_OPTIONS+=" -DDPLASMA_PRECISIONS=d"
USER_OPTIONS+=" -DPINS_ENABLE=ON"
USER_OPTIONS+=" -DDAGUE_PROF_TRACE=ON"
USER_OPTIONS+=" -DCORES_PER_SOCKET=6"

# These are override variables you can set (here or in the env) to alter defaults
CC=${CC:="/mnt/scratch/sw/intel/composer_xe_2013/bin/icc"}
CXX=${CXX:="/mnt/scratch/sw/intel/composer_xe_2013/bin/icpc"}
FC=${FC:="/mnt/scratch/sw/intel/composer_xe_2013/bin/ifort"}
MKL=${MKL:="/mnt/scratch/sw/intel/composer_xe_2013/mkl"}
CMAKE_C_FLAGS=${CMAKE_C_FLAGS:="-g3 -fPIC"}
CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS:="-g3 -fPIC"}
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:="$HOME/enthought_7.3.2_x64/"}
CMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH="$HOME/enthought_7.3.2_x64/"}
PLASMADIR=${PLASMA_DIR:="/home/bosilca/unstable/dplasma/PLASMA/build/"}
PAPI_DIR=${PAPI_DIR:="$HOME/sw/$HOSTNAME"}
HWLOC_DIR=${HWLOC_DIR:="/home/bosilca/opt/64"}
#MPI_DIR=${MPI_DIR:="/path/mpi"}
#HWLOC_DIR=${HWLOC_DIR:="/path/hwloc"}
#GTG_DIR=${GTG_DIR:="/path/gtg"}
#CUDA_DIR=${CUDA_DIR:="/path/cuda"}
#OMEGA_DIR=${OMEGA_DIR:="/path/omega"}
#PLASMA_DIR=${PLASMA_DIR:="/path/plasma"}

# This can be used to control auto detection of some packages
#PKG_CONFIG_PATH=/some/package/lib/pkgconfig:$PKG_CONFIG_PATH


. $(dirname $0)/config.inc
guess_defaults
run_cmake $*

# LOCATION=`dirname $0`
# echo ${LOCATION}
# export CC=
# export CXX=
# export F77=
# export MKL=
# export CFLAGS=
# export CMAKE_PREFIX_PATH=$HOME/enthought_7.3.2_x64/
# export CMAKE_FIND_ROOT_PATH=$HOME/enthought_7.3.2_x64/
# PLASMADIR="/home/bosilca/unstable/dplasma/PLASMA/build/"

# echo "cmake -G "Unix Makefiles" ./ -DPLASMA_DIR=${PLASMADIR} -DHWLOC_DIR=/home/bosilca/opt/64/ -DDPLASMA_SCHED_HWLOC=ON ${LOCATION} -DDAGUE_DIST_WITH_MPI=OFF -DPAPI_DIR=$HOME"
# cmake -G "Unix Makefiles" ./ -DPLASMA_DIR=${PLASMADIR} -DHWLOC_DIR=/home/bosilca/opt/64/ -DDPLASMA_SCHED_HWLOC=ON ${LOCATION} -DDAGUE_DIST_WITH_MPI=OFF -DPAPI_DIR=$HOME -DPYTHON_EXECUTABLE:FILEPATH=$HOME/enthought_7.3.2_x64/bin/python -DPYTHON_INCLUDE_DIRS:FILEPATH=$HOME/enthought_7.3.2_x64/include/ -DPYTHON_LIBRARIES:FILEPATH=$HOME/enthought_7.3.2_x64/lib/libpython2.7.so
