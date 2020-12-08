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
 * @file ParameterBlock.hpp
 * @brief Header file for the ParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_PARAMETERBLOCK_H_
#define INCLUDE_PARAMETERBLOCK_H_

#include <stdio.h>
#include <iostream>
#include <stdint.h>

#include <ceres/ceres.h>

/// @brief Base class providing the interface for parameter blocks.
class ParameterBlock {
 public:

  /// @brief Default constructor, assumes not fixed and no local parameterisation.
  ParameterBlock()
      : fixed_(false),
        localParameterizationPtr_(0) {
  }

  /// \brief Trivial destructor.
  virtual ~ParameterBlock() {
  }

  /// @name Setters
  /// @{



  /// @brief Directly set values of this parameter block.
  /// @param[in] parameters Pointer to parameters to be copied into this.
  virtual void setParameters(const double* parameters) = 0;

  /// @brief Whether or not this should be optimised at all.
  /// @param[in] fixed True if not to be optimised.
  void setFixed(bool fixed) {
    fixed_ = fixed;
  }

  /// @}

  /// @name Getters
  /// @{

  /// @brief Get parameter values.
  virtual double* parameters() = 0;

  /// @brief Get parameter values.
  virtual const double* parameters() const = 0;

  /// @brief Get the dimension of the parameter block.
  virtual size_t dimension() const = 0;

  /// @brief The dimension of the internal parameterisation (minimal representation).
  virtual size_t minimalDimension() const = 0;

  /// @brief Whether or not this is optimised at all.
  bool fixed() const {
    return fixed_;
  }

  virtual const ceres::LocalParameterization* localParameterizationPtr() const {
    return localParameterizationPtr_;
  }
  /// @}
  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const = 0;

 protected:

  bool fixed_;
  const ceres::LocalParameterization* localParameterizationPtr_;
};

#endif /* INCLUDE_PARAMETERBLOCK_H_ */