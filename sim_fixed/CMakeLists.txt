



add_executable(sim_fixed_app
  sim.cc
  sim.h
)

target_link_libraries(sim_fixed_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_opt_app
  sim_opt.cc
  sim.h
)

target_link_libraries(sim_fixed_opt_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_em_app
  sim_em.cc
  sim.h
)

target_link_libraries(sim_fixed_em_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_boem_app
  sim_boem.cc
  sim.h
)

target_link_libraries(sim_fixed_boem_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)