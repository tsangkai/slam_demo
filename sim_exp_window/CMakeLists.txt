

###
add_executable(sim_exp_win_app
  sim.h
  sim.cc
)

target_include_directories(sim_exp_win_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_exp_win_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_exp_win_opt_app
  sim.h
  sim_opt.cc
)

target_include_directories(sim_exp_win_opt_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_exp_win_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_exp_win_em_app
  sim.h
  sim_em.cc
)

target_include_directories(sim_exp_win_em_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_exp_win_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_exp_win_boem_app
  sim.h
  sim_boem.cc
)

target_include_directories(sim_exp_win_boem_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_exp_win_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)

