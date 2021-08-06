



add_executable(sim_fixed_app
  app.cc
  sim.h
)

target_link_libraries(sim_fixed_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE gflags::gflags
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_opt_app
  opt_app.cc
  sim.h
)

target_link_libraries(sim_fixed_opt_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE gflags::gflags
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_em_app
  em_app.cc
  sim.h
)

target_link_libraries(sim_fixed_em_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE gflags::gflags
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)



add_executable(sim_fixed_boem_app
  boem_app.cc
  sim.h
)

target_link_libraries(sim_fixed_boem_app
  PRIVATE ceres
  PRIVATE Eigen3::Eigen
  PRIVATE EigenRand
  PRIVATE gflags::gflags
  PRIVATE lib
  PRIVATE yaml-cpp::yaml-cpp
)