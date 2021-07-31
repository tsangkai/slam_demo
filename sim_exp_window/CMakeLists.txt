

add_executable(sim_exp_win_app
  sim.h
  sim.cc
)

target_link_libraries(sim_exp_win_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_exp_win_opt_app
  sim.h
  sim_opt.cc
)

target_link_libraries(sim_exp_win_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_exp_win_em_app
  sim.h
  sim_em.cc
)

target_link_libraries(sim_exp_win_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_exp_win_boem_app
  sim.h
  sim_boem.cc
)

target_link_libraries(sim_exp_win_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)

