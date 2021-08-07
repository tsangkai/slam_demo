
add_subdirectory(gflags)
add_subdirectory(yaml-cpp)









add_library(EigenRand
  EigenRand/EigenRand
  EigenRand/Core.h
  EigenRand/doc.h
  EigenRand/Macro.h
  EigenRand/MorePacketMath.h
  EigenRand/PacketFilter.h
  EigenRand/PacketRandomEngine.h
  EigenRand/RandUtils.h
  EigenRand/Dists/Basic.h
  EigenRand/Dists/Discrete.h
  EigenRand/Dists/GammaPoisson.h
  EigenRand/Dists/NormalExp.h
  EigenRand/MvDists/Multinomial.h
  EigenRand/MvDists/MvNormal.h
)

set_target_properties(EigenRand 
  PROPERTIES 
    LINKER_LANGUAGE C
)

target_include_directories(EigenRand
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(EigenRand
  PRIVATE Eigen3::Eigen
)
