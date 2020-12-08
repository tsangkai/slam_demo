
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

