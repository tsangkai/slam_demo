add_library(lib STATIC
  frontend.h
)

target_include_directories(lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(lib
  PRIVATE ${OpenCV_LIBS}
  PRIVATE ${EIGEN3_LIBS}
  PRIVATE ceres
)