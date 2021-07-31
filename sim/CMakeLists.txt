





add_executable(sim_app 
  sim.h
  sim_app.cc
)

target_link_libraries(sim_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC lib
)



add_executable(sim_opt_app 
  sim.h
  sim_opt_app.cc
)

target_link_libraries(sim_opt_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_em_app 
  sim.h
  sim_em_app.cc
)

target_link_libraries(sim_em_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC lib
)



###

add_executable(sim_boem_app 
  sim.h
  sim_boem_app.cc
)

target_link_libraries(sim_boem_app
  PUBLIC ${OpenCV_LIBS}
  PUBLIC Eigen3::Eigen
  PUBLIC EigenRand
  PUBLIC ceres
  PUBLIC lib
)



