





add_executable(sim_easy_app 
  sim.h
  app.cc
)

target_link_libraries(sim_easy_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_easy_opt_app 
  sim.h
  opt_app.cc
)

target_link_libraries(sim_easy_opt_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_easy_em_app 
  sim.h
  em_app.cc
)

target_link_libraries(sim_easy_em_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_easy_boem_app 
  sim.h
  boem_app.cc
)

target_link_libraries(sim_easy_boem_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



