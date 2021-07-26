



add_executable(sim_fixed_app
  sim.cc
)

target_include_directories(sim_fixed_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_fixed_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_fixed_opt_app
  sim_opt.cc
)

target_include_directories(sim_fixed_opt_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_fixed_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_fixed_em_app
  sim_em.cc
)

target_include_directories(sim_fixed_em_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_fixed_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_fixed_boem_app
  sim_boem.cc
)

target_include_directories(sim_fixed_boem_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_fixed_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)



