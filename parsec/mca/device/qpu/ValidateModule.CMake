# The QPU component needs nothing beyond what PaRSEC already requires: its
# only backend so far is an in-process mock. It is built on request only.

if( PARSEC_HAVE_QPU )
  set(MCA_${COMPONENT}_${MODULE} ON)
  file(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  set(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  set_property(TARGET parsec
               APPEND PROPERTY
                      PUBLIC_HEADER_H mca/device/qpu/device_qpu.h mca/device/qpu/device_qpu_qasm.h)
else( PARSEC_HAVE_QPU )
  message(STATUS "Module ${MODULE} not selectable: QPU support was not requested (PARSEC_DEVICE_WITH_QPU)")
  set(MCA_${COMPONENT}_${MODULE} OFF)
endif( PARSEC_HAVE_QPU )
