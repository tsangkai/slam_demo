

###

add_executable(sim_app_s_win
  sim.h
  sim_app_s_win.cc
)

target_link_libraries(sim_app_s_win
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_opt_app_s_win
  sim.h
  sim_opt_app_s_win.cc
)

target_link_libraries(sim_opt_app_s_win
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_em_app_s_win
  sim.h
  sim_em_app_s_win.cc
)

target_link_libraries(sim_em_app_s_win
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



###

add_executable(sim_boem_app_s_win
  sim.h
  sim_boem_app_s_win.cc
)

target_link_libraries(sim_boem_app_s_win
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)


