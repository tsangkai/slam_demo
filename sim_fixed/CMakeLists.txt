



add_executable(sim_fixed_app
  sim.cc
)

target_link_libraries(sim_fixed_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_fixed_opt_app
  sim_opt.cc
)

target_link_libraries(sim_fixed_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_fixed_em_app
  sim_em.cc
)

target_link_libraries(sim_fixed_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_fixed_boem_app
  sim_boem.cc
)

target_link_libraries(sim_fixed_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)