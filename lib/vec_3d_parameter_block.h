/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file PoseParameterBlock.hpp
 * @brief Header file for the PoseParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_VEC_3D_PARAMETERBLOCK_H_
#define INCLUDE_VEC_3D_PARAMETERBLOCK_H_

#include <Eigen/Core>

#include "sized_parameter_block.h"
// #include "PoseLocalParameterization.h"

/// \brief Wraps the parameter block for a pose estimate
class Vec3dParameterBlock: public SizedParameterBlock<3, 3, Eigen::Vector3d> {
 public:

  /// \brief The estimate type (3D vector).
  typedef Eigen::Vector3d estimate_t;

  /// \brief The base class type.
  typedef SizedParameterBlock<3, 3, estimate_t> base_t;

  /// \brief Default constructor (assumes not fixed).
  Vec3dParameterBlock(): 
    base_t::SizedParameterBlock() {
      setFixed(false);
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] T_WS The pose estimate as T_WS.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  Vec3dParameterBlock(const Eigen::Vector3d& point) {
    setEstimate(point);
    setFixed(false);
  }


  /// \brief Trivial destructor.
  ~Vec3dParameterBlock() {};

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] T_WS The estimate to set this to.
  void setEstimate(const Eigen::Vector3d& point) {
    // hack: only do "Euclidean" points for now...
    for (int i = 0; i < base_t::Dimension; ++i)
      parameters_[i] = point[i];
  }

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  Eigen::Vector3d estimate() const {
    return Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const {return "Vec3dParameterBlock";}

};


#endif /* INCLUDE_TIMED_3D_PARAMETERBLOCK_H_ */