

###

add_executable(sim_app 
  sim_app.cc
)

target_include_directories(sim_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_opt_app 
  sim_opt_app.cc
)

target_include_directories(sim_opt_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_em_app 
  sim_em_app.cc
)

target_include_directories(sim_em_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_boem_app 
  sim_boem_app.cc
)

target_include_directories(sim_boem_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



