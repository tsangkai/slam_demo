
###

add_executable(eigenRand_test 
  eigenRand_test.cc
)

target_include_directories(eigenRand_test
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
)

target_link_libraries(eigenRand_test
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
)



###

add_executable(reprojection_error_test 
  reprojection_error_test.cc
  transformation.h
)

target_include_directories(reprojection_error_test
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC lib
)

target_link_libraries(reprojection_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
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
  PUBLIC EigenRand
  PUBLIC lib
)

target_link_libraries(pre_int_imu_error_test
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC lib
)






















###

add_executable(reprojection_error_test_mc 
  reprojection_error_test_mc.cc
  transformation.h
)

target_include_directories(reprojection_error_test_mc
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC lib
)

target_link_libraries(reprojection_error_test_mc
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC lib
)




###

add_executable(pre_int_imu_error_test_mc
  pre_int_imu_error_test_mc.cc
  transformation.h
)

target_include_directories(pre_int_imu_error_test_mc
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC lib
)

target_link_libraries(pre_int_imu_error_test_mc
  PUBLIC ceres
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC lib
)