
###

add_executable(frontend_app frontend_app.cc)

target_include_directories(frontend_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC lib
)

target_link_libraries(frontend_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC lib
)


###

add_executable(backend_app backend_app.cc)

target_include_directories(backend_app
  PUBLIC ${OpenCV_INCLUDE_DIRS}          # not the target style
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${CERES_INCLUDE_DIRS}     
  PUBLIC lib
)

target_link_libraries(backend_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${EIGEN3_LIBS}
  PUBLIC ceres
  PUBLIC lib
)