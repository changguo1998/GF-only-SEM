// compress/include/gf/PrecisionPolicy.h
#pragma once
#include <hdf5.h>
#include <type_traits>

namespace gf {

/// Compile-time mapping from C++ type to HDF5 native type.
template <typename T>
struct hdf5_type;

template <>
struct hdf5_type<float> {
    static inline hid_t value() noexcept { return H5T_NATIVE_FLOAT; }
    static constexpr const char* label() noexcept { return "float32"; }
};

template <>
struct hdf5_type<double> {
    static inline hid_t value() noexcept { return H5T_NATIVE_DOUBLE; }
    static constexpr const char* label() noexcept { return "float64"; }
};

/// Runtime selection of HDF5 type ID.
/// Used when the precision choice is configurable at runtime.
inline hid_t select_precision_type(bool use_float32) noexcept {
    return use_float32 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
}

/// Size in bytes of the selected HDF5 type.
inline size_t precision_size(bool use_float32) noexcept {
    return use_float32 ? sizeof(float) : sizeof(double);
}

/// Human-readable label.
inline const char* precision_label(bool use_float32) noexcept {
    return use_float32 ? "float32" : "float64";
}

}  // namespace gf