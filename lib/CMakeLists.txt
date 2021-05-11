add_library(lib STATIC
  constant.h
  so3.h
  parameter_block.h
  sized_parameter_block.h
  vec_3d_parameter_block.h
  quat_parameter_block.h
  triangularization.h
  imu_data.h
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





#####

add_library(EigenRand STATIC
  EigenRand/EigenRand
  EigenRand/Core.h
  EigenRand/doc.h
  EigenRand/Macro.h
  EigenRand/MorePacketMath.h
  EigenRand/PacketFilter.h
  EigenRand/PacketRandomEngine.h
  EigenRand/RandUtils.h
  EigenRand/Dists/Basic.h
  EigenRand/Dists/Discrete.h
  EigenRand/Dists/GammaPoisson.h
  EigenRand/Dists/NormalExp.h
  EigenRand/MvDists/Multinomial.h
  EigenRand/MvDists/MvNormal.h
)

set_target_properties(EigenRand PROPERTIES LINKER_LANGUAGE C)

target_include_directories(EigenRand
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(EigenRand
  PRIVATE ${EIGEN3_LIBS}
)
