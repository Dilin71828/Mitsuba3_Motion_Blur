#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)

// WARNING: the AnimatedTransform class is outdated and dysfunctional with the
// latest version of Mitsuba 3. Please update this code before using it!
#if 0

AnimatedTransform::~AnimatedTransform() { }

void AnimatedTransform::append(const Keyframe &keyframe) {
    if (!m_keyframes.empty() && keyframe.time <= m_keyframes.back().time)
        Throw("AnimatedTransform::append(): time values must be "
              "strictly monotonically increasing!");

    if (m_keyframes.empty())
        m_transform = Transform4f(dr::transform_compose<Matrix4f>(
            keyframe.scale, keyframe.quat, keyframe.trans));

    m_keyframes.push_back(keyframe);
}

void AnimatedTransform::append(Float time, const Transform4f &trafo) {
    if (!m_keyframes.empty() && time <= m_keyframes.back().time)
        Throw("AnimatedTransform::append(): time values must be "
              "strictly monotonically increasing!");

    /* Perform a polar decomposition into a 3x3 scale/shear matrix,
       a rotation quaternion, and a translation vector. These will
       all be interpolated independently. */
    auto [M, Q, T] = dr::transform_decompose(trafo.matrix);

    if (m_keyframes.empty())
        m_transform = trafo;

    m_keyframes.push_back(Keyframe { time, M, Q, T });
}

bool AnimatedTransform::has_scale() const {
    if (m_keyframes.empty())
        return false;

    Matrix3f delta = dr::zeros<Matrix3f>();
    for (auto const &k: m_keyframes)
        delta += dr::abs(k.scale - dr::identity<Matrix3f>());
    return dr::sum_nested(delta) / m_keyframes.size() > 1e-3f;
}

typename AnimatedTransform::BoundingBox3f AnimatedTransform::translation_bounds() const {
    if (m_keyframes.empty()) {
        auto p = m_transform * Point3f(0.f);
        return BoundingBox3f(p, p);
    }
    Throw("AnimatedTransform::translation_bounds() not implemented for"
          " non-constant animation.");
}


std::ostream &operator<<(std::ostream &os, const AnimatedTransform::Keyframe &frame) {
    os << "Keyframe[" << std::endl
       << "  time = " << frame.time << "," << std::endl
       << "  scale = " << frame.scale << "," << std::endl
       << "  quat = " << frame.quat << "," << std::endl
       << "  trans = " << frame.trans
       << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const AnimatedTransform &t) {
    os << t.to_string();
    return os;
}

MI_IMPLEMENT_CLASS(AnimatedTransform, Object)
#endif


MI_VARIANT
AnimatedTransform<Float, Spectrum>::AnimatedTransform(const Properties &props) {
    m_time_start = props.get<ScalarFloat>("start", (ScalarFloat) 0.0f);
    m_time_end   = props.get<ScalarFloat>("end", (ScalarFloat) 1.0f);
    m_timestep_count = props.get<uint32_t>("timesteps", 1);
    for (auto &kv : props.transforms()) {
        m_transforms.push_back(kv.second);
    }
    initialize();
}

MI_VARIANT
AnimatedTransform<Float, Spectrum>::~AnimatedTransform() { }

MI_VARIANT
void AnimatedTransform<Float, Spectrum>::initialize() { 
    MI_IMPORT_CORE_TYPES()
    for (auto &v : m_transforms) {
        auto [M, Q, T] = dr::transform_decompose(v.matrix);
        m_scales.push_back(M);
        m_rotations.push_back(Q);
        m_translations.push_back(T);
    }
    m_timestep_length = (m_time_end - m_time_start) / (m_timestep_count - 1);

    m_scales_dr = dr::load<DynamicBuffer<Matrix3f>>(m_scales.data(), m_scales.size());
    m_rotations_dr = dr::load<DynamicBuffer<Quaternion4f>>(m_rotations.data(), m_rotations.size());
    m_translations_dr = dr::load<DynamicBuffer<Vector3f>>(m_translations.data(), m_translations.size());
}

MI_VARIANT
std::string AnimatedTransform<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "AnimatedTransform[]" << std::endl;

    return oss.str();
}

MI_VARIANT typename AnimatedTransform<Float, Spectrum>::ScalarTransform4f 
AnimatedTransform<Float, Spectrum>::get_transform_step(uint32_t timestep) const {
    return m_transforms[timestep];
}

MI_VARIANT typename AnimatedTransform<Float, Spectrum>::Transform4f 
AnimatedTransform<Float, Spectrum>::get_transform(Float time) const {
    MI_IMPORT_CORE_TYPES()

    if (likely(m_timestep_count <= 1))
        return Transform4f();

    UInt32 idx0 = dr::floor2int<UInt32, Float>((time - m_time_start) / m_timestep_length);
    UInt32 idx1 = idx0 + 1;

    // Compute relative time in [0,1]
    Float t0 = m_time_start + idx0 * m_timestep_length;
    Float t =
        dr::minimum(dr::maximum((time - t0) / m_timestep_length, 0.0f), 1.0f);

    // Interpolate scale, rotation and translation separately
    //Matrix3f M     = dr::gather<Matrix3f>(m_scales.data(), idx0) * (1 - t) +
    //                 dr::gather<Matrix3f>(m_scales.data(), idx1) * t;

    //Quaternion4f Q = dr::slerp(dr::gather<Quaternion4f>(m_rotations.data(), idx0),
    //                           dr::gather<Quaternion4f>(m_rotations.data(), idx1), t);
    //
    //Vector3f T     = dr::head<3>(dr::gather<Vector4f>(m_translations.data(), idx0)) * (1 - t) +
    //                 dr::head<3>(dr::gather<Vector4f>(m_translations.data(), idx1)) * t;

    Matrix3f M0 = dr::gather<Matrix3f>(m_scales_dr, idx0);
    Matrix3f M1 = dr::gather<Matrix3f>(m_scales_dr, idx1);
    Matrix3f M  = M0 * (1 - t) + M1 * t;
    
    Quaternion4f Q0 = dr::gather<Quaternion4f>(m_rotations_dr, idx0);
    Quaternion4f Q1 = dr::gather<Quaternion4f>(m_rotations_dr, idx1);
    Quaternion4f Q  = dr::slerp(Q0, Q1, t);
    
    Vector3f T0 = dr::gather<Vector3f>(m_translations_dr, idx0);
    Vector3f T1 = dr::gather<Vector3f>(m_translations_dr, idx1);
    Vector3f T  = T0 * (1 - t) + T1 * t;

    return Transform4f(dr::transform_compose<Matrix4f>(M, Q, T), 
        dr::transform_compose_inverse<Matrix4f>(M, Q, T));
}

MI_VARIANT typename AnimatedTransform<Float, Spectrum>::ScalarTransform4f 
AnimatedTransform<Float, Spectrum>::get_transform_scalar(ScalarFloat time) const {
    if (likely(m_timestep_count <= 1))
        return ScalarTransform4f();

    ScalarUInt32 idx0 = dr::floor2int<ScalarUInt32, ScalarFloat>(
        (time - m_time_start) / m_timestep_length);
    ScalarUInt32 idx1 = idx0 + 1;

    // Compute relative time in [0,1]
    ScalarFloat t0 = m_time_start + idx0 * m_timestep_length;
    ScalarFloat t =
        dr::minimum(dr::maximum((time - t0) / m_timestep_length, 0.0f), 1.0f);

    ScalarTransform4f transform0 = m_transforms[idx0];
    ScalarTransform4f transform1 = m_transforms[idx1];

    auto [M0, Q0, T0] = dr::transform_decompose(transform0.matrix);
    auto [M1, Q1, T1] = dr::transform_decompose(transform1.matrix);

    // Interpolate scale, rotation and translation separately
    ScalarMatrix3f M     = M0 * (1 - t) + M1 * t;
    ScalarQuaternion4f Q = dr::slerp(Q0, Q1, t);
    ScalarVector3f T     = T0 * (1 - t) + T1 * t;

    return ScalarTransform4f(
        dr::transform_compose<ScalarMatrix4f>(M, Q, T),
        dr::transform_compose_inverse<ScalarMatrix4f>(M, Q, T));
}

MI_IMPLEMENT_CLASS_VARIANT(AnimatedTransform, Object, "animated_transform")
MI_INSTANTIATE_CLASS(AnimatedTransform)
NAMESPACE_END(mitsuba)
