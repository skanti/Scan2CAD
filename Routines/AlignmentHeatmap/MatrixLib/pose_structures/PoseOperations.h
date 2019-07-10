#pragma once
#include "MatrixLib/matrix_structures/MatrixOperations.h"

namespace matrix_lib {
	
	/**
	 * Rotates the 3D point with rotation in SO3 (axis-angle) notation (i.e. 
	 * 3D vector).
	 */
	template<typename R, typename P>
	 auto rotatePointWithSO3(const R& rotation, const P& point) {
		using T = typename BaseType<R, P>::type;
		using RType = typename ResultType<R, P>::type;

		const auto omega = makeTuple(rotation[I<0>()], rotation[I<1>()], rotation[I<2>()]);
		const auto p = makeTuple(point[I<0>()], point[I<1>()], point[I<2>()]);
		const auto theta2 = dot3(omega, omega);
		
		Tuple<typename TL<RType, RType, RType>::type> rotatedPoint;
		if (theta2 > T(FLT_EPSILON)) {
			// Away from zero, use the Rodrigues' formula
			//
			//   result = point costheta +
			//            (w x point) * sintheta +
			//            w (w . point) (1 - costheta)
			//
			// We want to be careful to only evaluate the square root if the
			// norm of the angle_axis vector is greater than zero. Otherwise
			// we get a division by zero.

			const auto theta = sqrt(theta2);
			const auto cosTheta = cos(theta);
			const auto sinTheta = sin(theta);
			const auto thetaInverse = T(1) / theta;

			const auto w = scale3(omega, thetaInverse);
			const auto wCrossPoint = cross3(w, p);
			const auto tmp = dot3(w, p) * (T(1) - cosTheta);

			auto rotatedPointTuple = add3(
				add3(
					scale3(p, cosTheta), 
					scale3(wCrossPoint, sinTheta)
				),
				scale3(w, tmp)
			);
			
			rotatedPoint[I<0>()] = rotatedPointTuple[I<0>()];
			rotatedPoint[I<1>()] = rotatedPointTuple[I<1>()];
			rotatedPoint[I<2>()] = rotatedPointTuple[I<2>()];
		}
		else {
			// Near zero, the first order Taylor approximation of the rotation
			// matrix R corresponding to a vector w and angle w is
			//
			//   R = I + hat(w) * sin(theta)
			//
			// But sintheta ~ theta and theta * w = angle_axis, which gives us
			//
			//  R = I + hat(w)
			//
			// and actually performing multiplication with the point pt, gives us
			// R * pt = pt + w x pt.
			//
			// Switching to the Taylor expansion near zero provides meaningful
			// derivatives when evaluated using Jets.

			const auto wCrossPoint = cross3(omega, p);
			auto rotatedPointTuple = add3(p, wCrossPoint);

			rotatedPoint[I<0>()] = rotatedPointTuple[I<0>()];
			rotatedPoint[I<1>()] = rotatedPointTuple[I<1>()];
			rotatedPoint[I<2>()] = rotatedPointTuple[I<2>()];
		}

		return rotatedPoint;
	}


	/**
	 * Rotates the 3D point with rotation given as unit quaternion notation (i.e.
	 * 4D vector (real, imag0, imag1, imag2)).
	 */
	template<typename R, typename P>
	 auto rotatePointWithUnitQuaternion(const R& rotation, const P& point) {
		using T = typename BaseType<R, P>::type;

		// p = 0 + px * i + py * j + pz * k
		// pRotated = q * p * qInverse
		const auto p0 = point[I<0>()];
		const auto p1 = point[I<1>()];
		const auto p2 = point[I<2>()];

		const auto real = rotation[I<0>()];
		const auto imag0 = rotation[I<1>()];
		const auto imag1 = rotation[I<2>()];
		const auto imag2 = rotation[I<3>()];

		const auto t2 = real * imag0;
		const auto t3 = real * imag1;
		const auto t4 = real * imag2;
		const auto t5 = -imag0 * imag0;
		const auto t6 = imag0 * imag1;
		const auto t7 = imag0 * imag2;
		const auto t8 = -imag1 * imag1;
		const auto t9 = imag1 * imag2;
		const auto t1 = -imag2 * imag2;

		return makeTuple(
			T(2) * ((t8 + t1) * p0 + (t6 - t4) * p1 + (t3 + t7) * p2) + p0,
			T(2) * ((t4 + t6) * p0 + (t5 + t1) * p1 + (t9 - t2) * p2) + p1,
			T(2) * ((t7 - t3) * p0 + (t2 + t9) * p1 + (t5 + t8) * p2) + p2
		);
	}


	/**
	 * Converts a SO3 3D vector to 3x3 rotation matrix (with row-wise memory storage).
	 */
	template<typename P>
	 auto convertSO3ToMatrix(const P& omega) {
		using T = typename BaseType<P>::type;
		using RType = typename ResultType<P>::type;

		// We use the Rodrigues' formula for exponential map.
		// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

		auto theta2 = dot3(omega, omega);
		auto omegaHat = makeTuple(
			T(0), -omega[I<2>()], omega[I<1>()],
			omega[I<2>()], T(0), -omega[I<0>()],
			-omega[I<1>()], omega[I<0>()], T(0)
		);
		
		auto id3x3 = makeTuple(
			T(1), T(0), T(0),
			T(0), T(1), T(0),
			T(0), T(0), T(1)
		);

		// We need to be careful to not divide by zero. 
		Tuple<typename AddElements<9, RType, NullType>::type> rotationMatrix;
		if (theta2 > T(FLT_EPSILON)) {
			// Rodrigues' formula.
			const auto theta = sqrt(theta2);
			auto m = add3x3(
				id3x3, 
				add3x3(
					scale3x3(omegaHat, (sin(theta) / theta)),
					scale3x3(mm3x3(omegaHat, omegaHat), ((T(1) - cos(theta)) / theta2))
				)
			);
			assign3x3(m, rotationMatrix);
		}
		else {
			// If theta squared is too small, we use only a first order approximation of the exponential formula.
			// R = exp(omega) = I + omega_hat + omega_hat^2 / 2! + ...
			auto m = add3x3(id3x3, omegaHat);
			assign3x3(m, rotationMatrix);
		}

		return rotationMatrix;
	}


	/**
	 * Converts a unit quaternion 4D vector to 3x3 rotation matrix (with row-wise memory storage).
	 */
	template<typename P>
	 auto convertUnitQuaternionToMatrix(const P& q) {
		using T = typename BaseType<P>::type;

		auto real = q[I<0>()];
		auto imag0 = q[I<1>()];
		auto imag1 = q[I<2>()];
		auto imag2 = q[I<3>()];
		
		return makeTuple(
			real * real + imag0 * imag0 - imag1 * imag1 - imag2 * imag2,
			T(2) * (imag0 * imag1 - real * imag2),
			T(2) * (imag0 * imag2 + real * imag1),
			T(2) * (imag0 * imag1 + real * imag2),
			real * real - imag0 * imag0 + imag1 * imag1 - imag2 * imag2,
			T(2) * (imag1 * imag2 - real * imag0),
			T(2) * (imag0 * imag2 - real * imag1),
			T(2) * (imag1 * imag2 + real * imag0),
			real * real - imag0 * imag0 - imag1 * imag1 + imag2 * imag2
		);	
	}


	/**
	 * Methods for applying rotation with compile-time decision about the rotation type.
	 */
	template<typename R, typename P, unsigned DeformationFlag>
	 auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<DeformationFlag>);

	template<typename R, typename P>
	 auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::AFFINE>) {
		return mv3x3(rotation, point);
	}

	template<typename R, typename P>
	 auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::SE3>) {
		return rotatePointWithSO3(rotation, point);
	}

	template<typename R, typename P>
	 auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::QUATERNION>) {
		return rotatePointWithUnitQuaternion(rotation, point);
	}


	/**
	 * Methods for extracting a particular part of the pose (rotation or translation) from the
	 * pose array. They are useful also for making a copy of global data into the local memory.
	 */
	template<typename Pose, unsigned DeformationFlag>
	 auto extractRotation(const Pose& pose, Unsigned2Type<DeformationFlag>);

	template<typename Pose>
	 auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::AFFINE>) {
		return makeTuple(
			pose[I<0>()], pose[I<1>()], pose[I<2>()],
			pose[I<3>()], pose[I<4>()], pose[I<5>()],
			pose[I<6>()], pose[I<7>()], pose[I<8>()]
		);
	}

	template<typename Pose>
	 auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::SE3>) {
		return makeTuple(pose[I<0>()], pose[I<1>()], pose[I<2>()]);
	}

	template<typename Pose>
	 auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::QUATERNION>) {
		return makeTuple(pose[I<0>()], pose[I<1>()], pose[I<2>()], pose[I<3>()]);
	}

	template<typename Pose, unsigned DeformationFlag>
	 auto extractTranslation(const Pose& pose, Unsigned2Type<DeformationFlag>);

	template<typename Pose>
	 auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::AFFINE>) {
		return makeTuple(pose[I<9>()], pose[I<10>()], pose[I<11>()]);
	}

	template<typename Pose>
	 auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::SE3>) {
		return makeTuple(pose[I<3>()], pose[I<4>()], pose[I<5>()]);
	}

	template<typename Pose>
	 auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::QUATERNION>) {
		return makeTuple(pose[I<4>()], pose[I<5>()], pose[I<6>()]);
	}

	
	/**
	 * Methods for applying rotation with compile-time decision about the rotation type.
	 */
	template<typename RotationVec, unsigned DeformationFlag>
	 auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<DeformationFlag>);

	template<typename RotationVec>
	 auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::AFFINE>) {
		// The rotation vector is already a matrix.
		return rotationVec;
	}

	template<typename RotationVec>
	 auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::QUATERNION>) {
		return convertUnitQuaternionToMatrix(rotationVec);
	}

	template<typename RotationVec>
	 auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::SE3>) {
		return convertSO3ToMatrix(rotationVec);
	}


	/**
	 * Method for applying pose with compile-time decision about the rotation type.
	 */
	template<typename Pose, typename P, unsigned DeformationFlag>
	 auto transformPoint(const Pose& pose, const P& point, Unsigned2Type<DeformationFlag>) {
		const auto rotation = extractRotation(pose, Unsigned2Type<DeformationFlag>());
		const auto translation = extractTranslation(pose, Unsigned2Type<DeformationFlag>());

		return add3(
			rotatePoint(rotation, point, Unsigned2Type<DeformationFlag>()),
			translation
		);
	}

} // namespace matrix_lib 
