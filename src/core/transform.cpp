#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)

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
    //for (uint32_t i = 0; i < m_transforms.size(); i++) {
    //    std::cout << m_scales[i] << '\n';
    //}
    //for (uint32_t i = 0; i < m_rotations.size(); i++) {
    //    std::cout << m_rotations[i] << '\n';
    //}
    //for (uint32_t i = 0; i < m_translations.size(); i++) {
    //    std::cout << m_translations[i] << '\n';
    //}

    //if constexpr (dr::is_jit_v<Float>) {
        m_scales_rearranged.resize(9);
        m_rotations_rearranged.resize(4);
        m_translations_rearranged.resize(3);
        std::unique_ptr<float[]> buffer(new float[m_timestep_count]);
        for (uint32_t x = 0; x < 3; x++) {
            for (uint32_t y = 0; y < 3; y++) {
                float *buffer_ptr = buffer.get();
                for (uint32_t step = 0; step < m_timestep_count; step++) {
                    dr::store(buffer_ptr, m_scales[step].entry(x, y));
                    m_scales_rearranged[x * 3 + y].push_back(
                        dr::load<ScalarFloat>(buffer_ptr, 1));
                    buffer_ptr += 1;
                }
                m_scales_dr.entry(x, y) =
                    dr::load<Float>(buffer.get(), m_timestep_count);
            }
        }
        for (uint32_t x = 0; x < 4; x++) {
            float *buffer_ptr = buffer.get();
            for (uint32_t step = 0; step < m_timestep_count; step++) {
                dr::store(buffer_ptr, m_rotations[step].entry(x));
                m_rotations_rearranged[x].push_back(
                    dr::load<ScalarFloat>(buffer_ptr, 1));
                buffer_ptr += 1;
            }
            m_rotations_dr.entry(x) =
                dr::load<Float>(buffer.get(), m_timestep_count);
        }
        for (uint32_t x = 0; x < 3; x++) {
            float *buffer_ptr = buffer.get();
            for (uint32_t step = 0; step < m_timestep_count; step++) {
                dr::store(buffer_ptr, m_translations[step].entry(x));
                m_translations_rearranged[x].push_back(
                    dr::load<ScalarFloat>(buffer_ptr, 1));
                buffer_ptr += 1;
            }
            m_translations_dr.entry(x) =
                dr::load<Float>(buffer.get(), m_timestep_count);
        }
    //}
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

    //std::cout << t << '\n';

    // Interpolate scale, rotation and translation separately
    if constexpr (dr::is_array_v<Float>) {      
        Matrix3f M0 = dr::gather<Matrix3f>(m_scales_dr, idx0);
        Matrix3f M1 = dr::gather<Matrix3f>(m_scales_dr, idx1);
        Matrix3f M  = M0 * (1 - t) + M1 * t;
        //std::cout << M << '\n';

        Quaternion4f Q0, Q1;
        for (uint32_t x = 0; x < 4; x++) {
            Q0.entry(x) = dr::gather<Float>(m_rotations_dr.entry(x), idx0);
            Q1.entry(x) = dr::gather<Float>(m_rotations_dr.entry(x), idx1);
        }
        //Quaternion4f Q0 = dr::gather<Quaternion4f>(m_rotations_dr, idx0);
        //Quaternion4f Q1 = dr::gather<Quaternion4f>(m_rotations_dr, idx1);
        Quaternion4f Q  = dr::slerp(Q0, Q1, t);
        //std::cout << Q0 << '\n' << Q1 << '\n' << Q << '\n';

        Vector3f T0 = dr::gather<Vector3f>(m_translations_dr, idx0);
        Vector3f T1 = dr::gather<Vector3f>(m_translations_dr, idx1);
        Vector3f T  = T0 * (1 - t) + T1 * t;
        //std::cout << T << '\n';

        return Transform4f(dr::transform_compose<Matrix4f>(M, Q, T));
    } else {
        Matrix3f M = m_scales[idx0] * (1 - t) + m_scales[idx1] * t;
    
        Quaternion4f Q = dr::slerp(m_rotations[idx0], m_rotations[idx1], t);
    
        Vector3f T = m_translations[idx0]*(1 - t) + m_translations[idx1] * t;
    
        return Transform4f(dr::transform_compose<Matrix4f>(M, Q, T));
    }
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

    return ScalarTransform4f(dr::transform_compose<ScalarMatrix4f>(M, Q, T));
}

MI_IMPLEMENT_CLASS_VARIANT(AnimatedTransform, Object, "animated_transform")
MI_INSTANTIATE_CLASS(AnimatedTransform)
NAMESPACE_END(mitsuba)
