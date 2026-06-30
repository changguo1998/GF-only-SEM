#ifndef GF_VTK_WRITER_HH
#define GF_VTK_WRITER_HH

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

// Legacy VTK v3.0 binary unstructured-grid writer.
// All floating-point data stored big-endian float32.
// All integer data stored big-endian int32.
//
// Thread-safe: each call writes to a separate file handle.

namespace gf_vtk {

inline void write_big_endian(FILE *fp, const void *data, size_t nbytes) {
    fwrite(data, 1, nbytes, fp);
}

// Swap bytes for big-endian output on little-endian hosts.
template <typename T>
static void to_big_endian(T *val) {
    union { T v; unsigned char b[sizeof(T)]; } u;
    u.v = *val;
    for (size_t i = 0; i < sizeof(T) / 2; ++i) {
        unsigned char tmp = u.b[i];
        u.b[i] = u.b[sizeof(T) - 1 - i];
        u.b[sizeof(T) - 1 - i] = tmp;
    }
    *val = u.v;
}

// Convert float to big-endian int32 bytes.
inline float to_big_float(float val) {
    to_big_endian(&val);
    return val;
}
inline int32_t to_big_int32(int32_t val) {
    to_big_endian(&val);
    return val;
}

class VtkWriter {
  public:
    VtkWriter(const std::string &path, const std::string &title)
        : fp_(std::fopen(path.c_str(), "wb")) {
        if (!fp_) {
            throw std::runtime_error("Cannot open " + path);
        }
        fprintf(fp_, "# vtk DataFile Version 3.0\n%s\nBINARY\nDATASET UNSTRUCTURED_GRID\n",
                title.c_str());
    }

    ~VtkWriter() {
        if (fp_) std::fclose(fp_);
    }

    VtkWriter(const VtkWriter &) = delete;
    VtkWriter &operator=(const VtkWriter &) = delete;

    void write_points(const std::vector<float> &coords) {
        int64_t n = coords.size() / 3;
        fprintf(fp_, "POINTS %ld float\n", (long)n);
        // Convert to big-endian float32
        std::vector<float> be(coords.size());
        for (size_t i = 0; i < coords.size(); ++i) be[i] = to_big_float(coords[i]);
        fwrite(be.data(), sizeof(float), be.size(), fp_);
        fputc('\n', fp_);
    }

    // cells: flattened [n_verts, v0, v1, ..., vk] for each cell
    // cell_types: VTK cell type per cell
    void write_cells(const std::vector<int32_t> &cells,
                     const std::vector<int32_t> &cell_types) {
        int64_t n_cells = cell_types.size();
        fprintf(fp_, "CELLS %ld %ld\n", (long)n_cells, (long)cells.size());
        std::vector<int32_t> be(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) be[i] = to_big_int32(cells[i]);
        fwrite(be.data(), sizeof(int32_t), be.size(), fp_);
        fputc('\n', fp_);

        fprintf(fp_, "CELL_TYPES %ld\n", (long)n_cells);
        std::vector<int32_t> be_types(n_cells);
        for (int64_t i = 0; i < n_cells; ++i) be_types[i] = to_big_int32(cell_types[i]);
        fwrite(be_types.data(), sizeof(int32_t), be_types.size(), fp_);
        fputc('\n', fp_);
    }

    void begin_cell_data(int64_t n_cells) {
        fprintf(fp_, "CELL_DATA %ld\n", (long)n_cells);
    }

    void write_scalar_field(const std::string &name, const std::vector<float> &data) {
        fprintf(fp_, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name.c_str());
        std::vector<float> be(data.size());
        for (size_t i = 0; i < data.size(); ++i) be[i] = to_big_float(data[i]);
        fwrite(be.data(), sizeof(float), be.size(), fp_);
        fputc('\n', fp_);
    }

    void begin_point_data(int64_t n_points) {
        fprintf(fp_, "POINT_DATA %ld\n", (long)n_points);
    }

  private:
    FILE *fp_ = nullptr;
};

} // namespace gf_vtk

#endif // GF_VTK_WRITER_HH