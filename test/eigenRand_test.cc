#include <iostream>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
 
using namespace Eigen;
 
int main()
{
  // Initialize random number generator with seed=42 for following codes.
  // Or you can use C++11 RNG such as std::mt19937 or std::ranlux48.
  Rand::Vmt19937_64 urng{ 42 };
 
  // this will generate 4x4 real matrix with range [-1, 1]
  MatrixXf mat = Rand::balanced<MatrixXf>(4, 4, urng);
  std::cout << mat << std::endl;
 
  // this will generate 10x10 real 2d array on the normal distribution
  ArrayXXf arr = Rand::normal<ArrayXXf>(10, 10, urng);
  std::cout << arr << std::endl;
 
  Vector3d vec = Rand::normal<Vector3d>(3, 1, urng);
  std::cout << vec << std::endl;

  return 0;
}