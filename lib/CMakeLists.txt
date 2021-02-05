add_library(lib STATIC
  so3.h
  parameter_block.h
  sized_parameter_block.h
  vec_3d_parameter_block.h
  quat_parameter_block.h
  triangularization.h
  imu_data.h
  imu_error.h
  pre_int_imu_error.h
  reprojection_error.h
)

target_include_directories(lib
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(lib
  PRIVATE ${OpenCV_LIBS}
  PRIVATE ${EIGEN3_LIBS}
  PRIVATE ceres
)