





add_executable(app 
  sim.h
  app.cc
)

target_link_libraries(app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(opt_app 
  sim.h
  opt_app.cc
)

target_link_libraries(opt_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(em_app 
  sim.h
  em_app.cc
)

target_link_libraries(em_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(boem_app 
  sim.h
  boem_app.cc
)

target_link_libraries(boem_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



