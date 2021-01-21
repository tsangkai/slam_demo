
###

add_executable(reprojection_error_test 
  reprojection_error_test.cc
  transformation.h
)

target_include_directories(reprojection_error_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC lib
)

target_link_libraries(reprojection_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC lib
)



###

add_executable(imu_error_test 
  imu_error_test.cc
  transformation.h
)

target_include_directories(imu_error_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC lib
)

target_link_libraries(imu_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC lib
)


###

add_executable(pre_int_imu_error_test 
  pre_int_imu_error_test.cc
  transformation.h
)

target_include_directories(pre_int_imu_error_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC lib
)

target_link_libraries(pre_int_imu_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC lib
)