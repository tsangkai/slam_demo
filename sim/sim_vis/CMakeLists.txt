

###

add_executable(sim_opt_vis
        sim_opt_vis.cc
)

target_include_directories(sim_opt_vis
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_opt_vis
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_em_vis
        sim_em_vis.cc
)

target_include_directories(sim_em_vis
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_em_vis
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)



###

add_executable(sim_boem_vis
        sim_boem_vis.cc
)

target_include_directories(sim_boem_vis
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC lib
)

target_link_libraries(sim_boem_vis
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC lib
)

add_executable(sim_vis
        sim_vis.cc
        )

target_include_directories(sim_vis
        PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
        PUBLIC ${EIGEN3_INCLUDE_DIR}
        PUBLIC ${CERES_INCLUDE_DIRS}
        PUBLIC ${Boost_INCLUDE_DIRS}
        PUBLIC lib
        )

target_link_libraries(sim_vis
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${EIGEN3_LIBS}
        PUBLIC ceres
        PUBLIC ${Boost_LIBRARIES}
        PUBLIC lib
        )

add_executable(sim_opt_win
        sim_opt_win.cc
        )

target_include_directories(sim_opt_win
        PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
        PUBLIC ${EIGEN3_INCLUDE_DIR}
        PUBLIC ${CERES_INCLUDE_DIRS}
        PUBLIC ${Boost_INCLUDE_DIRS}
        PUBLIC lib
        )

target_link_libraries(sim_opt_win
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${EIGEN3_LIBS}
        PUBLIC ceres
        PUBLIC ${Boost_LIBRARIES}
        PUBLIC lib
        )



###

add_executable(sim_em_win
        sim_em_win.cc
        )

target_include_directories(sim_em_win
        PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
        PUBLIC ${EIGEN3_INCLUDE_DIR}
        PUBLIC ${CERES_INCLUDE_DIRS}
        PUBLIC ${Boost_INCLUDE_DIRS}
        PUBLIC lib
        )

target_link_libraries(sim_em_win
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${EIGEN3_LIBS}
        PUBLIC ceres
        PUBLIC ${Boost_LIBRARIES}
        PUBLIC lib
        )



###

add_executable(sim_boem_win
        sim_boem_win.cc
        )

target_include_directories(sim_boem_win
        PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
        PUBLIC ${EIGEN3_INCLUDE_DIR}
        PUBLIC ${CERES_INCLUDE_DIRS}
        PUBLIC ${Boost_INCLUDE_DIRS}
        PUBLIC lib
        )

target_link_libraries(sim_boem_win
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${EIGEN3_LIBS}
        PUBLIC ceres
        PUBLIC ${Boost_LIBRARIES}
        PUBLIC lib
        )
