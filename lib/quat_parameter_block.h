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

#ifndef INCLUDE_QUAT_PARAMETERBLOCK_H_
#define INCLUDE_QUAT_PARAMETERBLOCK_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sized_parameter_block.h"
// #include "PoseLocalParameterization.h"


/// \brief Wraps the parameter block for a pose estimate
class QuatParameterBlock: public SizedParameterBlock<4, 3, Eigen::Quaterniond> {
 public:

  /// \brief The estimate type (3D vector).
  typedef Eigen::Quaterniond estimate_t;

  /// \brief The base class type.
  typedef SizedParameterBlock<4, 3, Eigen::Quaterniond> base_t;

  /// \brief Default constructor (assumes not fixed).
  QuatParameterBlock(): 
    base_t::SizedParameterBlock() {
      setFixed(false);
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] T_WS The pose estimate as T_WS.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  QuatParameterBlock(const Eigen::Quaterniond& quat) {
    setEstimate(quat);
    setFixed(false);
  }

  /// \brief Trivial destructor.
  ~QuatParameterBlock() {}

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] T_WS The estimate to set this to.
  void setEstimate(const Eigen::Quaterniond& quat) {
      parameters_[0] = quat.w();
      parameters_[1] = quat.x();
      parameters_[2] = quat.y();
      parameters_[3] = quat.z();
  }

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  Eigen::Quaterniond estimate() const {
    return Eigen::Quaterniond(parameters_[0], parameters_[1], parameters_[2], parameters_[3]);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const {
    return "QuatParameterBlock";
  }

};


#endif /* INCLUDE_QUAT_PARAMETERBLOCK_H_ */