




###

add_executable(sim_easy_app
  sim_easy_app.cc
)

target_include_directories(sim_easy_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_easy_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_easy_opt_app
  sim_easy_opt_app.cc
)

target_include_directories(sim_easy_opt_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_easy_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)




###

add_executable(sim_easy_em_app
  sim_easy_em_app.cc
)

target_include_directories(sim_easy_em_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_easy_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_easy_boem_app
  sim_easy_boem_app.cc
)

target_include_directories(sim_easy_boem_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC EigenRand
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_easy_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



