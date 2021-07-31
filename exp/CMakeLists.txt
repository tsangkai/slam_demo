

add_executable(gt_process_app 
  gt_process_app.cc
)

target_link_libraries(gt_process_app
  PUBLIC Eigen3::Eigen
)



add_executable(traj_process_app 
  traj_process_app.cc
)

target_link_libraries(traj_process_app
  PUBLIC Eigen3::Eigen
)



add_executable(frontend_app 
  frontend_app.cc
  frontend.h
)

target_link_libraries(frontend_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC lib
)



add_executable(backend_opt_app 
  backend_opt_app.cc
  backend.h
)

target_link_libraries(backend_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)







add_executable(backend_em_app 
  backend_em_app.cc
)

target_link_libraries(backend_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)



add_executable(backend_boem_app 
  backend_boem_app.cc
)

target_link_libraries(backend_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC ceres
  PUBLIC lib
)