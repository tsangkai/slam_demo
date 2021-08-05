

###

add_executable(sim_sliding_window_app
  sim.h
  app.cc
)

target_link_libraries(sim_sliding_window_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_sliding_window_opt_app
  sim.h
  opt_app.cc
)

target_link_libraries(sim_sliding_window_opt_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_sliding_window_em_app
  sim.h
  em_app.cc
)

target_link_libraries(sim_sliding_window_em_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_sliding_window_boem_app
  sim.h
  boem_app.cc
)

target_link_libraries(sim_sliding_window_boem_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)


