#pragma once

#if defined(__GNUG__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdouble-promotion"
#elif defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdouble-promotion"
#endif

#include <mitsuba/core/ray.h>
#include <drjit/transform.h>
#include <drjit/sphere.h>

NAMESPACE_BEGIN(mitsuba)
/**
 * \brief Encapsulates a 4x4 homogeneous coordinate transformation along with
 * its inverse transpose
 *
 * The Transform class provides a set of overloaded matrix-vector
 * multiplication operators for vectors, points, and normals (all of them
 * behave differently under homogeneous coordinate transformations, hence
 * the need to represent them using separate types)
 */
template <typename Point_> struct Transform {

    // =============================================================
    //! @{ \name Type declarations
    // =============================================================

    static constexpr size_t Size = Point_::Size;

    using Float   = dr::value_t<Point_>;
    using Matrix  = dr::Matrix<Float, Size>;
    using Mask    = dr::mask_t<Float>;
    using Scalar  = dr::scalar_t<Float>;

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Fields
    // =============================================================

    Matrix matrix            = dr::identity<Matrix>();
    Matrix inverse_transpose = dr::identity<Matrix>();

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Constructors, methods, etc.
    // =============================================================

    /// Initialize the transformation from the given matrix (and compute its inverse transpose)
    Transform(const Matrix &value)
        : matrix(value),
          inverse_transpose(dr::inverse_transpose(value)) { }

    /// Initialize the transformation from the given matrix and its inverse
    Transform(const Matrix &value, const Matrix &inv)
        : matrix(value),
          inverse_transpose(inv) { }

    /// Concatenate transformations
    MI_INLINE Transform operator*(const Transform &other) const {
        return Transform(matrix * other.matrix,
                         inverse_transpose * other.inverse_transpose);
    }

    /// Compute the inverse of this transformation (involves just shuffles, no arithmetic)
    MI_INLINE Transform inverse() const {
        return Transform(transpose(inverse_transpose), transpose(matrix));
    }

    /// Get the translation part of a matrix
    Vector<Float, Size - 1> translation() const {
        return dr::head<Size - 1>(matrix.entry(Size - 1));
    }

    /// Equality comparison operator
    bool operator==(const Transform &t) const {
        return matrix == t.matrix &&
               inverse_transpose == t.inverse_transpose;
    }

    /// Inequality comparison operator
    bool operator!=(const Transform &t) const {
        return matrix != t.matrix ||
               inverse_transpose != t.inverse_transpose;
    }

    /**
     * \brief Transform a 3D vector/point/normal/ray by a transformation that
     * is known to be an affine 3D transformation (i.e. no perspective)
     */
    template <typename T>
    MI_INLINE auto transform_affine(const T &input) const {
        return operator*(input);
    }

    /// Transform a point (handles affine/non-perspective transformations only)
    template <typename T, typename Expr = dr::expr_t<Float, T>>
    MI_INLINE Point<Expr, Size - 1> transform_affine(const Point<T, Size - 1> &arg) const {
        dr::Array<Expr, Size> result = matrix.entry(Size - 1);

        DRJIT_UNROLL for (size_t i = 0; i < Size - 1; ++i)
            result = dr::fmadd(matrix.entry(i), arg.entry(i), result);

        return dr::head<Size - 1>(result); // no-op
    }

    /**
     * \brief Transform a 3D point
     * \remark In the Python API, one should use the \c @ operator
     */
    template <typename T, typename Expr = dr::expr_t<Float, T>>
    MI_INLINE Point<Expr, Size - 1> operator*(const Point<T, Size - 1> &arg) const {
        dr::Array<Expr, Size> result = matrix.entry(Size - 1);

        DRJIT_UNROLL for (size_t i = 0; i < Size - 1; ++i)
            result = dr::fmadd(matrix.entry(i), arg.entry(i), result);

        return dr::head<Size - 1>(result) / result.entry(Size - 1);
    }

    /**
     * \brief Transform a 3D vector
     * \remark In the Python API, one should use the \c @ operator
     */
    template <typename T, typename Expr = dr::expr_t<Float, T>>
    MI_INLINE Vector<Expr, Size - 1> operator*(const Vector<T, Size - 1> &arg) const {
        dr::Array<Expr, Size> result = matrix.entry(0);
        result *= arg.x();

        DRJIT_UNROLL for (size_t i = 1; i < Size - 1; ++i)
            result = dr::fmadd(matrix.entry(i), arg.entry(i), result);

        return dr::head<Size - 1>(result); // no-op
    }

    /**
     * \brief Transform a 3D normal vector
     * \remark In the Python API, one should use the \c @ operator
     */
    template <typename T, typename Expr = dr::expr_t<Float, T>>
    MI_INLINE Normal<Expr, Size - 1> operator*(const Normal<T, Size - 1> &arg) const {
        dr::Array<Expr, Size> result = inverse_transpose.entry(0);
        result *= arg.x();

        DRJIT_UNROLL for (size_t i = 1; i < Size - 1; ++i)
            result = dr::fmadd(inverse_transpose.entry(i), arg.entry(i), result);

        return dr::head<Size - 1>(result); // no-op
    }

    /// Transform a ray (for perspective transformations)
    template <typename T, typename Spectrum, typename Expr = dr::expr_t<Float, T>,
              typename Result = Ray<Point<Expr, Size - 1>, Spectrum>>
    MI_INLINE Result operator*(const Ray<Point<T, Size - 1>, Spectrum> &ray) const {
        return Result(operator*(ray.o), operator*(ray.d), ray.maxt, ray.time,
                      ray.wavelengths);
    }

    /// Transform a ray (for affine/non-perspective transformations)
    template <typename T, typename Spectrum, typename Expr = dr::expr_t<Float, T>,
              typename Result = Ray<Point<Expr, Size - 1>, Spectrum>>
    MI_INLINE Result transform_affine(const Ray<Point<T, Size - 1>, Spectrum> &ray) const {
        return Result(transform_affine(ray.o), transform_affine(ray.d),
                      ray.maxt, ray.time, ray.wavelengths);
    }

    /// Create a translation transformation
    static Transform translate(const Vector<Float, Size - 1> &v) {
        return Transform(dr::translate<Matrix>(v),
                         transpose(dr::translate<Matrix>(-v)));
    }

    /// Create a scale transformation
    static Transform scale(const Vector<Float, Size - 1> &v) {
        return Transform(dr::scale<Matrix>(v),
                         // No need to transpose a diagonal matrix.
                         dr::scale<Matrix>(dr::rcp(v)));
    }

    /// Create a rotation transformation around an arbitrary axis in 3D. The angle is specified in degrees
    template <size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform rotate(const Vector<Float, Size - 1> &axis, const Float &angle) {
        Matrix matrix = dr::rotate<Matrix>(axis, dr::deg_to_rad(angle));
        return Transform(matrix, matrix);
    }

    /// Create a rotation transformation in 2D. The angle is specified in degrees
    template <size_t N = Size, dr::enable_if_t<N == 3> = 0>
    static Transform rotate(const Float &angle) {
        Matrix matrix = dr::rotate<Matrix>(dr::deg_to_rad(angle));
        return Transform(matrix, matrix);
    }

    /** \brief Create a perspective transformation.
     *   (Maps [near, far] to [0, 1])
     *
     *  Projects vectors in camera space onto a plane at z=1:
     *
     *  x_proj = x / z
     *  y_proj = y / z
     *  z_proj = (far * (z - near)) / (z * (far-near))
     *
     *  Camera-space depths are not mapped linearly!
     *
     * \param fov Field of view in degrees
     * \param near Near clipping plane
     * \param far  Far clipping plane
     */
    template <size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform perspective(Float fov, Float near_, Float far_) {
        Float recip = 1.f / (far_ - near_);

        /* Perform a scale so that the field of view is mapped
           to the interval [-1, 1] */
        Float tan = dr::tan(dr::deg_to_rad(fov * .5f)),
              cot = 1.f / tan;

        Matrix trafo = dr::diag(Vector<Float, Size>(cot, cot, far_ * recip, 0.f));
        trafo(2, 3) = -near_ * far_ * recip;
        trafo(3, 2) = 1.f;

        Matrix inv_trafo = dr::diag(Vector<Float, Size>(tan, tan, 0.f, dr::rcp(near_)));
        inv_trafo(2, 3) = 1.f;
        inv_trafo(3, 2) = (near_ - far_) / (far_ * near_);

        return Transform(trafo, transpose(inv_trafo));
    }

    /** \brief Create an orthographic transformation, which maps Z to [0,1]
     * and leaves the X and Y coordinates untouched.
     *
     * \param near Near clipping plane
     * \param far  Far clipping plane
     */
    template <size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform orthographic(Float near_, Float far_) {
        return scale({1.f, 1.f, 1.f / (far_ - near_)}) *
               translate({ 0.f, 0.f, -near_ });
    }

    /** \brief Create a look-at camera transformation
     *
     * \param origin Camera position
     * \param target Target vector
     * \param up     Up vector
     */
    template <size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform look_at(const Point<Float, 3> &origin,
                             const Point<Float, 3> &target,
                             const Vector<Float, 3> &up) {
        using Vector1 = dr::Array<Scalar, 1>;
        using Vector3 = Vector<Float, 3>;

        Vector3 dir    = dr::normalize(target - origin);
        Vector3 left   = dr::normalize(dr::cross(up, dir));
        Vector3 new_up = dr::cross(dir, left);

        Vector1 z(0);
        Matrix result = Matrix(
            dr::concat(left, z),
            dr::concat(new_up, z),
            dr::concat(dir, z),
            dr::concat(origin, Vector1(1))
        );

        Matrix inverse = dr::transpose(Matrix(
            dr::concat(left, z),
            dr::concat(new_up, z),
            dr::concat(dir, z),
            Vector<Float, 4>(0.f, 0.f, 0.f, 1.f)
        ));

        inverse[3] = inverse * dr::concat(-origin, Vector1(1));

        return Transform(result, dr::transpose(inverse));
    }

    /// Creates a transformation that converts from the standard basis to 'frame'
    template <typename Value, size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform to_frame(const Frame<Value> &frame) {
        dr::Array<Scalar, 1> z(0);
        Matrix result = Matrix(
            dr::concat(frame.s, z),
            dr::concat(frame.t, z),
            dr::concat(frame.n, z),
            Vector<Float, 4>(0.f, 0.f, 0.f, 1.f)
        );

        return Transform(result, result);
    }

    /// Creates a transformation that converts from 'frame' to the standard basis
    template <typename Value, size_t N = Size, dr::enable_if_t<N == 4> = 0>
    static Transform from_frame(const Frame<Value> &frame) {
        dr::Array<Scalar, 1> z(0);
        Matrix result = dr::transpose(Matrix(
            dr::concat(frame.s, z),
            dr::concat(frame.t, z),
            dr::concat(frame.n, z),
            Vector<Float, 4>(0.f, 0.f, 0.f, 1.f)
        ));

        return Transform(result, result);
    }

    //! @}
    // =============================================================


    // =============================================================
    //! @{ \name Test for transform properties.
    // =============================================================

    /**
     * \brief Test for a scale component in each transform matrix by checking
     * whether <tt>M . M^T == I</tt> (where <tt>M</tt> is the matrix in
     * question and <tt>I</tt> is the identity).
     */
    Mask has_scale() const {
        Mask mask(false);
        for (size_t i = 0; i < Size - 1; ++i) {
            for (size_t j = i; j < Size - 1; ++j) {
                Float sum = 0.f;
                for (size_t k = 0; k < Size - 1; ++k)
                    sum += matrix[i][k] * matrix[j][k];

                mask |= dr::abs(sum - (i == j ? 1.f : 0.f)) > 1e-3f;
            }
        }
        return mask;
    }

    /// Extract a lower-dimensional submatrix
    template <size_t ExtractedSize = Size - 1,
              typename Result = Transform<Point<Float, ExtractedSize>>>
    MI_INLINE Result extract() const {
        Result result;
        for (size_t i = 0; i < ExtractedSize - 1; ++i) {
            for (size_t j = 0; j < ExtractedSize - 1; ++j) {
                result.matrix.entry(i, j) = matrix.entry(i, j);
                result.inverse_transpose.entry(i, j) =
                    inverse_transpose.entry(i, j);
            }
            result.matrix.entry(ExtractedSize - 1, i) =
                matrix.entry(Size - 1, i);
            result.inverse_transpose.entry(i, ExtractedSize - 1) =
                inverse_transpose.entry(i, Size - 1);
        }

        result.matrix.entry(ExtractedSize - 1, ExtractedSize - 1) =
            matrix.entry(Size - 1, Size - 1);

        result.inverse_transpose.entry(ExtractedSize - 1, ExtractedSize - 1) =
            inverse_transpose.entry(Size - 1, Size - 1);

        return result;
    }

    //! @}
    // =============================================================

    DRJIT_STRUCT(Transform, matrix, inverse_transpose)
};

// WARNING: the AnimatedTransform class is outdated and dysfunctional with the
// latest version of Mitsuba 3. Please update this code before using it!
/**
 * \brief Encapsulates an animated 4x4 homogeneous coordinate transformation
 *
 * The animation is stored as keyframe animation with linear segments. The
 * implementation performs a polar decomposition of each keyframe into a 3x3
 * scale/shear matrix, a rotation quaternion, and a translation vector. These
 * will all be interpolated independently at eval time.
 .. tabs::
    .. code-tab:: xml
        :name: ...

        <shape type="...">
            <animated_transform>
                <scale x="2" y="10" z="1"/>
            </animated_transform>
        </shape>
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB AnimatedTransform : public Object {
public:
    MI_IMPORT_CORE_TYPES()

    /// Instantiate a animated transform from a \ref Properties object
    AnimatedTransform(const Properties &props);

    /// Return the number of timesteps
    uint32_t timestep_count() const { return m_timestep_count; }

    /// Return the start time
    ScalarFloat time_start() const { return m_time_start; }
    /// Return the end time
    ScalarFloat time_end() const { return m_time_end; }

    /**
     * \brief Return an interpolated transformation at given time
     * \param A queried timestep which should lie between \ref m_time_start
     * and \ref m_time_end
    */
    Transform4f get_transform(Float time) const;

    template<typename FloatP>
    Transform<Point<FloatP, 4>> get_transform_packet(FloatP time) const {
        using UInt32P = dr::uint32_array_t<FloatP>;
        using Transform4fP = Transform<Point<FloatP, 4>>;
        using Matrix3fP = dr::Matrix<FloatP, 3>;
        using Matrix4fP = dr::Matrix<FloatP, 4>;
        using Quaternion4fP = dr::Quaternion<FloatP>;
        using Vector3fP = Vector<FloatP, 3>;
        using Vector4fP = Vector<FloatP, 4>;

        if (likely(m_timestep_count <= 1))
            return Transform4fP();

        UInt32P idx0 = dr::floor2int<UInt32P, FloatP>((time - m_time_start) /
                                                   m_timestep_length);
        UInt32P idx1 = idx0 + 1;

        // Compute relative time in [0,1]
        //std::cout << time << '\n';
        FloatP t0 = m_time_start + idx0 * m_timestep_length;
        FloatP t  = dr::minimum(dr::maximum((time - t0) / m_timestep_length, 0.0f), 1.0f);

        //std::cout << t << '\n';

        // Interpolate scale, rotation and translation separately
        if constexpr (dr::is_jit_v<FloatP>) {
            Matrix3fP M0 = dr::gather<Matrix3fP>(m_scales_dr, idx0);
            Matrix3fP M1 = dr::gather<Matrix3fP>(m_scales_dr, idx1);
            Matrix3fP M  = M0 * (1 - t) + M1 * t;

            //std::cout << M0 << '\n' << M1 << '\n' << M << '\n';

            Quaternion4fP Q0 = dr::gather<Quaternion4fP>(m_rotations_dr, idx0);
            Quaternion4fP Q1 = dr::gather<Quaternion4fP>(m_rotations_dr, idx1);
            Quaternion4fP Q  = dr::slerp(Q0, Q1, t);

            Vector3fP T0 = dr::gather<Vector3fP>(m_translations_dr, idx0);
            Vector3fP T1 = dr::gather<Vector3fP>(m_translations_dr, idx1);
            Vector3fP T  = T0 * (1 - t) + T1 * t;

            return Transform4fP(dr::transform_compose<Matrix4fP>(M, Q, T));
        } else if constexpr (dr::is_array_v<FloatP>) 
        {
            Matrix3fP M;
            for (uint32_t x = 0; x < 3; x++) {
                for (uint32_t y = 0; y < 3; y++) {
                    M.entry(x, y) =
                        dr::gather<FloatP>(m_scales_rearranged[x * 3 + y].data(), idx0) * (1 - t) +
                        dr::gather<FloatP>(m_scales_rearranged[x * 3 + y].data(), idx1) * t;
                }            
            }

            Quaternion4fP Q0, Q1, Q;
            for (uint32_t x = 0; x < 4; x++) {
                Q0.entry(x) =
                    dr::gather<FloatP>(m_rotations_rearranged[x].data(), idx0);
                Q1.entry(x) =
                    dr::gather<FloatP>(m_rotations_rearranged[x].data(), idx1);
            }
            Q = dr::slerp(Q0, Q1, t);

            Vector3fP T;
            for (uint32_t x = 0; x < 3; x++) {
                T.entry(x) =
                    dr::gather<FloatP>(m_translations_rearranged[x].data(), idx0) * (1 - t) +
                    dr::gather<FloatP>(m_translations_rearranged[x].data(), idx1) * t;
            }

            return Transform4fP(dr::transform_compose<Matrix4fP>(M, Q, T));

        } 
        else {
            Matrix3fP M = m_scales[idx0] * (1 - t) + m_scales[idx1] * t;
        
            Quaternion4fP Q = dr::slerp(m_rotations[idx0], m_rotations[idx1], t);
        
            Vector3fP T = m_translations[idx0] * (1 - t) + m_translations[idx1] * t;
        
            return Transform4fP(dr::transform_compose<Matrix4fP>(M, Q, T));
        }
    }

    /**
     * \brief Return transformation according to a given timestep.
     * \param A queried timestep which should lie between 0 to m_timestep_count - 1
    */
    ScalarTransform4f get_transform_step(uint32_t timestep) const;

    ScalarTransform4f get_transform_scalar(ScalarFloat time) const;


    /// Return a human-readable summary of this object
    virtual std::string to_string() const override;

    MI_DECLARE_CLASS()
protected:
    //AnimatedTransform(const Properties &props);
    virtual ~AnimatedTransform();
    void initialize();

protected:
    uint32_t m_timestep_count;
    ScalarFloat m_time_start;
    ScalarFloat m_time_end;
    ScalarFloat m_timestep_length;
    std::vector<ScalarTransform4f> m_transforms;
    std::vector<ScalarMatrix3f> m_scales;
    std::vector<ScalarQuaternion4f> m_rotations;
    std::vector<ScalarVector3f> m_translations;

    Matrix3f m_scales_dr;
    Quaternion4f m_rotations_dr;
    Vector3f m_translations_dr;
    
    std::vector<std::vector<ScalarFloat>> m_scales_rearranged;
    std::vector<std::vector<ScalarFloat>> m_rotations_rearranged;
    std::vector<std::vector<ScalarFloat>> m_translations_rearranged;
};

MI_EXTERN_CLASS(AnimatedTransform)

// -----------------------------------------------------------------------
//! @{ \name Printing
// -----------------------------------------------------------------------

template <typename Point>
std::ostream &operator<<(std::ostream &os, const Transform<Point> &t) {
    os << t.matrix;
    return os;
}

// std::ostream &operator<<(std::ostream &os, const AnimatedTransform::Keyframe &frame);
// std::ostream &operator<<(std::ostream &os, const AnimatedTransform &t);

//! @}
// -----------------------------------------------------------------------

#if defined(__GNUG__)
#  pragma GCC diagnostic pop
#elif defined(__clang__)
#  pragma clang diagnostic pop
#endif

NAMESPACE_END(mitsuba)
