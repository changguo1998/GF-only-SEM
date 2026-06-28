/* preprocess/cpp/main.cpp — C++ accelerated preprocessor
 *
 * Standalone executable (no MPI).  Reads mesh.h5 topology + CLI params,
 * computes GLL geometry, CFL timestep, and PML damping, then writes
 * results back to mesh.h5.
 *
 * Usage:
 *   gf_preprocess_cpp <mesh.h5> <N> <cfl_safety> \
 *       <pml_xmin> <pml_xmax> <pml_ymin> <pml_ymax> <pml_zmin> <pml_zmax>
 *
 * Sets field/element/{coords,dxi_dx,jacobian,mass,damping} and
 * field/info/solver_dt in mesh.h5.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hdf5.h>
#include <hdf5_hl.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// HDF5 helpers
// -----------------------------------------------------------------------

static hid_t open_or_fail(const char* path, unsigned flags) {
    hid_t fid = H5Fopen(path, flags, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    return fid;
}

static std::vector<int64_t> read_int64_1d(hid_t loc, const char* name, hsize_t& n_out) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: cannot open dataset %s\n", name);
        exit(1);
    }
    hid_t space = H5Dget_space(ds);
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_out = 1;
    for (int d = 0; d < ndims; ++d)
        n_out *= dims[d];
    std::vector<int64_t> buf(n_out);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

static void read_attr_double(hid_t loc, const char* name, double& val) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) {
        val = 0.0;
        return;
    }
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
}

// Write a 5-D double array (n_cell, NGLL, NGLL, NGLL, last_dim)
static void write_5d_double(hid_t loc, const char* name, hsize_t n_cell, hsize_t NGLL,
                            hsize_t last_dim, const double* data) {
    hsize_t dims[5] = {n_cell, NGLL, NGLL, NGLL, last_dim};
    hid_t space = H5Screate_simple(5, dims, nullptr);
    hid_t ds =
        H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(space);
}

// Write a 4-D double array (n_cell, NGLL, NGLL, NGLL)
static void write_4d_double(hid_t loc, const char* name, hsize_t n_cell, hsize_t NGLL,
                            const double* data) {
    hsize_t dims[4] = {n_cell, NGLL, NGLL, NGLL};
    hid_t space = H5Screate_simple(4, dims, nullptr);
    hid_t ds =
        H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(space);
}

// Write a scalar attribute
static void write_scalar_attr(hid_t loc, const char* name, double val) {
    if (H5Aexists(loc, name))
        H5Adelete(loc, name);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        return;
    }
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    H5Sclose(space);
}

// -----------------------------------------------------------------------
// GLL quadrature
// -----------------------------------------------------------------------

static void gll_quadrature(int N, std::vector<double>& pts, std::vector<double>& w) {
    int ngll = N + 1;
    pts.resize(ngll);
    w.resize(ngll);

    if (N == 0) {
        pts[0] = 0.0;
        w[0] = 2.0;
        return;
    }
    if (N == 1) {
        pts[0] = -1.0;
        pts[1] = 1.0;
        w[0] = 1.0;
        w[1] = 1.0;
        return;
    }

    // Legendre polynomial P_N(x) via recurrence
    // Get its derivative roots (GLL interior points)
    // We use the standard approach: roots of (1-x^2) * P'_N(x)
    // Build companion matrix of P'_N and compute eigenvalues

    // P'_N coefficients (derivative of Legendre)
    // P_N(x) = sum_{k=0}^{N} c_k x^k
    // We compute roots of P'_N(x) = 0 using the Jacobi matrix approach
    // For GLL nodes: -1, roots of P'_N, 1

    // Use the tridiagonal Jacobi matrix for Legendre polynomials
    // Beta_j = j / sqrt(4*j^2 - 1) for j=1..N-1
    // Eigenvalues of this (N-1)x(N-1) matrix are the interior GLL nodes
    int n_int = N - 1;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_int, n_int);
    for (int j = 1; j <= n_int; ++j) {
        double beta = j / std::sqrt(4.0 * j * j - 1.0);
        if (j > 1)
            J(j - 1, j - 2) = beta;
        J(j - 1, j - 1) = 0.0;
        if (j < n_int)
            J(j - 1, j) = beta;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(J);
    Eigen::VectorXd evals = eigensolver.eigenvalues();

    // Sort eigenvalues
    std::sort(evals.data(), evals.data() + n_int);

    // Concatenate: -1, interior roots, +1
    pts[0] = -1.0;
    for (int i = 0; i < n_int; ++i)
        pts[i + 1] = evals(i);
    pts[ngll - 1] = 1.0;

    // Compute weights: w_i = 2 / (N*(N+1) * [P_N(x_i)]^2)
    // Evaluate P_N(x) via recurrence
    for (int i = 0; i < ngll; ++i) {
        double x = pts[i];
        // Recurrence for P_N
        double P0 = 1.0;
        double P1 = x;
        for (int k = 2; k <= N; ++k) {
            double Pk = ((2.0 * k - 1.0) * x * P1 - (k - 1.0) * P0) / k;
            P0 = P1;
            P1 = Pk;
        }
        double PN = (N == 0) ? 1.0 : (N == 1) ? x : P1;
        w[i] = 2.0 / (N * (N + 1) * PN * PN);
    }
}

// -----------------------------------------------------------------------
// GLL geometry for a single element
// -----------------------------------------------------------------------

// Reference hex corners in [-1,1]^3
static const double HEX_CORNERS[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

// Linear shape function values and derivatives at (xi, eta, zeta)
static inline void linear_shape(double xi, double eta, double zeta, double N_vals[8],
                                double dN[8][3]) {
    for (int a = 0; a < 8; ++a) {
        double ca = HEX_CORNERS[a][0];
        double cb = HEX_CORNERS[a][1];
        double cc = HEX_CORNERS[a][2];
        double t = 0.125;
        N_vals[a] = t * (1.0 + ca * xi) * (1.0 + cb * eta) * (1.0 + cc * zeta);
        dN[a][0] = t * ca * (1.0 + cb * eta) * (1.0 + cc * zeta);
        dN[a][1] = t * (1.0 + ca * xi) * cb * (1.0 + cc * zeta);
        dN[a][2] = t * (1.0 + ca * xi) * (1.0 + cb * eta) * cc;
    }
}

// -----------------------------------------------------------------------
// Topology data loaded from HDF5
// -----------------------------------------------------------------------
struct Topology {
    int64_t n_cell, n_surface, n_edge, n_vertex;
    std::vector<int64_t> cell_to_surface;  // [n_cell, 6]
    std::vector<int64_t> surface_to_edge;  // [n_surface, 4]
    std::vector<int64_t> edge_to_vertex;   // [n_edge, 2]
    std::vector<double> vertex_to_coord;   // [n_vertex, 3]
};

static Topology read_topology(const char* mesh_path) {
    hid_t fid = open_or_fail(mesh_path, H5F_ACC_RDONLY);
    hid_t topo_gid = H5Gopen2(fid, "topology", H5P_DEFAULT);
    if (topo_gid < 0) {
        fprintf(stderr, "ERROR: /topology/ group not found\n");
        exit(1);
    }

    Topology topo;
    // Read attributes
    topo.n_cell = 0;
    topo.n_surface = 0;
    topo.n_edge = 0;
    topo.n_vertex = 0;
    {
        hsize_t tmp;
        hid_t attr;
        attr = H5Aopen(topo_gid, "n_cell", H5P_DEFAULT);
        if (attr >= 0) {
            H5Aread(attr, H5T_NATIVE_INT64, &topo.n_cell);
            H5Aclose(attr);
        }
        attr = H5Aopen(topo_gid, "n_surface", H5P_DEFAULT);
        if (attr >= 0) {
            H5Aread(attr, H5T_NATIVE_INT64, &topo.n_surface);
            H5Aclose(attr);
        }
        attr = H5Aopen(topo_gid, "n_edge", H5P_DEFAULT);
        if (attr >= 0) {
            H5Aread(attr, H5T_NATIVE_INT64, &topo.n_edge);
            H5Aclose(attr);
        }
        attr = H5Aopen(topo_gid, "n_vertex", H5P_DEFAULT);
        if (attr >= 0) {
            H5Aread(attr, H5T_NATIVE_INT64, &topo.n_vertex);
            H5Aclose(attr);
        }
    }

    // If attributes weren't set, infer from datasets
    hsize_t n;
    topo.cell_to_surface = read_int64_1d(topo_gid, "cell_to_surface", n);
    if (topo.n_cell == 0)
        topo.n_cell = static_cast<int64_t>(n / 6);
    topo.surface_to_edge = read_int64_1d(topo_gid, "surface_to_edge", n);
    if (topo.n_surface == 0)
        topo.n_surface = static_cast<int64_t>(n / 4);
    topo.edge_to_vertex = read_int64_1d(topo_gid, "edge_to_vertex", n);
    if (topo.n_edge == 0)
        topo.n_edge = static_cast<int64_t>(n / 2);

    // Read vertex_to_coord as 2D
    {
        hid_t ds = H5Dopen2(topo_gid, "vertex_to_coord", H5P_DEFAULT);
        hid_t space = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        topo.n_vertex = static_cast<int64_t>(dims[0]);
        topo.vertex_to_coord.resize(static_cast<size_t>(dims[0] * 3));
        H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, topo.vertex_to_coord.data());
        H5Dclose(ds);
        H5Sclose(space);
    }

    H5Gclose(topo_gid);
    H5Fclose(fid);
    return topo;
}

// -----------------------------------------------------------------------
// GLL vertex IDs for one element (GMSH hex ordering)
// -----------------------------------------------------------------------
static void get_cell_vertex_ids(int64_t e,
                                const int64_t* c2s,  // [n_cell, 6]
                                const int64_t* s2e,  // [n_surface, 4]
                                const int64_t* e2v,  // [n_edge, 2]
                                int64_t* vertex_ids  // [8] output
) {
    // Collect face vertices for all 6 faces of element e
    int64_t face_verts[6][8];
    int face_counts[6] = {0, 0, 0, 0, 0, 0};

    for (int fi = 0; fi < 6; ++fi) {
        int64_t signed_sid = c2s[e * 6 + fi];
        int64_t abs_sid = (signed_sid < 0) ? -signed_sid - 1 : signed_sid - 1;
        for (int ei = 0; ei < 4; ++ei) {
            int64_t sedge = s2e[abs_sid * 4 + ei];
            int64_t abs_eid = (sedge < 0) ? -sedge - 1 : sedge - 1;
            int64_t v0 = e2v[abs_eid * 2 + 0];
            int64_t v1 = e2v[abs_eid * 2 + 1];
            // Add unique vertices
            bool has0 = false, has1 = false;
            for (int k = 0; k < face_counts[fi]; ++k) {
                if (face_verts[fi][k] == v0)
                    has0 = true;
                if (face_verts[fi][k] == v1)
                    has1 = true;
            }
            if (!has0 && face_counts[fi] < 8)
                face_verts[fi][face_counts[fi]++] = v0;
            if (!has1 && face_counts[fi] < 8)
                face_verts[fi][face_counts[fi]++] = v1;
        }
    }

    // GMSH corner → face membership
    // Face order: 0=-z, 1=+z, 2=-y, 3=+y, 4=-x, 5=+x
    static const int corner_faces[8][3] = {
        {0, 2, 4},  // v0: -z, -y, -x
        {0, 2, 5},  // v1: -z, -y, +x
        {0, 3, 5},  // v2: -z, +y, +x
        {0, 3, 4},  // v3: -z, +y, -x
        {1, 2, 4},  // v4: +z, -y, -x
        {1, 2, 5},  // v5: +z, -y, +x
        {1, 3, 5},  // v6: +z, +y, +x
        {1, 3, 4},  // v7: +z, +y, -x
    };

    // For each target face set, find the vertex that appears on exactly those faces
    // We'll build vertex → face set mapping first
    // Since vertices may repeat across faces, use a different strategy:
    // For each corner, find the unique vertex belonging to all 3 target faces
    for (int c = 0; c < 8; ++c) {
        int tf0 = corner_faces[c][0];
        int tf1 = corner_faces[c][1];
        int tf2 = corner_faces[c][2];

        // Intersection of the 3 face vertex sets
        int64_t found = -1;
        for (int k0 = 0; k0 < face_counts[tf0]; ++k0) {
            int64_t v = face_verts[tf0][k0];
            // Check if v is in face tf1
            bool in_tf1 = false;
            for (int k1 = 0; k1 < face_counts[tf1]; ++k1) {
                if (face_verts[tf1][k1] == v) {
                    in_tf1 = true;
                    break;
                }
            }
            if (!in_tf1)
                continue;
            // Check if v is in face tf2
            bool in_tf2 = false;
            for (int k2 = 0; k2 < face_counts[tf2]; ++k2) {
                if (face_verts[tf2][k2] == v) {
                    in_tf2 = true;
                    break;
                }
            }
            if (in_tf2) {
                found = v;
                break;
            }
        }
        if (found < 0) {
            fprintf(stderr, "ERROR: cannot identify corner %d for element %lld\n", c,
                    (long long)e);
            exit(1);
        }
        vertex_ids[c] = found;
    }
}

// -----------------------------------------------------------------------
// Main computation: GLL geometry, CFL min spacing, PML damping
// -----------------------------------------------------------------------
struct ComputeResult {
    std::vector<double> coords;    // [n_cell, NGLL, NGLL, NGLL, 3]
    std::vector<double> dxi_dx;    // [n_cell, NGLL, NGLL, NGLL, 9]
    std::vector<double> jacobian;  // [n_cell, NGLL, NGLL, NGLL]
    std::vector<double> mass;      // [n_cell, NGLL, NGLL, NGLL]
    std::vector<double> damping;   // [n_cell, NGLL, NGLL, NGLL]
    double cfl_dt;                 // CFL-limited timestep
};

static ComputeResult compute_all(
    const Topology& topo, int N, double cfl_safety,
    double pml_thickness[6],       // xmin, xmax, ymin, ymax, zmin, zmax
    const double domain_bounds[6]  // xmin, xmax, ymin, ymax, zmin, zmax
) {
    int64_t n_cell = topo.n_cell;
    int ngll = N + 1;
    int64_t n_node = ngll * ngll * ngll;
    int64_t n_total = n_cell * n_node;

    ComputeResult res;
    res.coords.resize(n_total * 3, 0.0);
    res.dxi_dx.resize(n_total * 9, 0.0);
    res.jacobian.resize(n_total, 0.0);
    res.mass.resize(n_total, 0.0);
    res.damping.resize(n_total, 0.0);
    res.cfl_dt = 1e30;

    // GLL quadrature points and weights
    std::vector<double> pts, w;
    gll_quadrature(N, pts, w);

    // Domain extents for PML width estimate
    double dx_ext = domain_bounds[1] - domain_bounds[0];
    double dy_ext = domain_bounds[3] - domain_bounds[2];
    double dz_ext = domain_bounds[5] - domain_bounds[4];
    double n_cells_axis = std::max(1.0, std::pow(static_cast<double>(n_cell), 1.0 / 3.0));
    double cell_size_est = std::max({dx_ext, dy_ext, dz_ext}) / n_cells_axis;

    // Temporary storage for corner coordinates
    double cv[8][3];

    // vp_max for CFL (we don't have vp yet — user provides as Python callable)
    // Instead we compute h_min only and let Python combine with vp_max
    double h_min = 1e30;

    // Precompute full GLL weights product w[i]*w[j]*w[k] per node offset
    // (used for mass matrix)
    std::vector<double> w3(ngll * ngll * ngll);
    for (int i = 0; i < ngll; ++i)
        for (int j = 0; j < ngll; ++j)
            for (int k = 0; k < ngll; ++k)
                w3[(i * ngll + j) * ngll + k] = w[i] * w[j] * w[k];

    // Read is_pml from mesh.h5 if it exists (written by Python boundary_detector)
    // Otherwise compute it later from PML thickness (we'll skip per-element PML check)

    // We'll compute damping without is_pml flag — the Python side sets it.
    // Here we compute the spatial ramp for ALL elements, Python multiplies by is_pml.
    // PML ramp parameters
    double pml_width[6];
    for (int f = 0; f < 6; ++f)
        pml_width[f] = pml_thickness[f] * cell_size_est;
    double pml_start[6];  // PML entry (interior face)
    double pml_end[6];    // domain boundary
    // faces: 0=xmin,1=xmax,2=ymin,3=ymax,4=zmin,5=zmax
    pml_start[0] = domain_bounds[0] + pml_width[0];
    pml_end[0] = domain_bounds[0];
    pml_start[1] = domain_bounds[1] - pml_width[1];
    pml_end[1] = domain_bounds[1];
    pml_start[2] = domain_bounds[2] + pml_width[2];
    pml_end[2] = domain_bounds[2];
    pml_start[3] = domain_bounds[3] - pml_width[3];
    pml_end[3] = domain_bounds[3];
    pml_start[4] = domain_bounds[4] + pml_width[4];
    pml_end[4] = domain_bounds[4];
    pml_start[5] = domain_bounds[5] - pml_width[5];
    pml_end[5] = domain_bounds[5];

    const int64_t* c2s = topo.cell_to_surface.data();
    const int64_t* s2e = topo.surface_to_edge.data();
    const int64_t* e2v = topo.edge_to_vertex.data();
    const double* v2c = topo.vertex_to_coord.data();

    // Precompute vertex IDs and corner coords for all elements (single-threaded)
    struct ElemCorners {
        int64_t ids[8];
        double coords[8][3];
    };
    std::unique_ptr<ElemCorners[]> elem_data(new ElemCorners[n_cell]);
    for (int64_t e = 0; e < n_cell; ++e) {
        get_cell_vertex_ids(e, c2s, s2e, e2v, elem_data[e].ids);
        for (int vi = 0; vi < 8; ++vi) {
            int64_t vid = elem_data[e].ids[vi] - 1;
            elem_data[e].coords[vi][0] = v2c[vid * 3 + 0];
            elem_data[e].coords[vi][1] = v2c[vid * 3 + 1];
            elem_data[e].coords[vi][2] = v2c[vid * 3 + 2];
        }
    }

#pragma omp parallel for schedule(dynamic, 1) reduction(min : h_min) \
    shared(res, elem_data, pts, w, w3, pml_width) firstprivate(ngll)
    for (int64_t e = 0; e < n_cell; ++e) {
        const double (*cv)[3] = elem_data[e].coords;

        // Element bounding box (from precomputed corner coords)
        double e_xmin = 1e30, e_xmax = -1e30;
        double e_ymin = 1e30, e_ymax = -1e30;
        double e_zmin = 1e30, e_zmax = -1e30;
        for (int vi = 0; vi < 8; ++vi) {
            if (cv[vi][0] < e_xmin)
                e_xmin = cv[vi][0];
            if (cv[vi][0] > e_xmax)
                e_xmax = cv[vi][0];
            if (cv[vi][1] < e_ymin)
                e_ymin = cv[vi][1];
            if (cv[vi][1] > e_ymax)
                e_ymax = cv[vi][1];
            if (cv[vi][2] < e_zmin)
                e_zmin = cv[vi][2];
            if (cv[vi][2] > e_zmax)
                e_zmax = cv[vi][2];
        }

        // Loop over all GLL nodes
        for (int i = 0; i < ngll; ++i) {
            double xi = pts[i];
            for (int j = 0; j < ngll; ++j) {
                double eta = pts[j];
                for (int k = 0; k < ngll; ++k) {
                    double zeta = pts[k];

                    // Linear shape functions and derivatives
                    double N_vals[8], dN[8][3];
                    linear_shape(xi, eta, zeta, N_vals, dN);

                    // Physical coordinates
                    double x_phys[3] = {0, 0, 0};
                    for (int a = 0; a < 8; ++a) {
                        x_phys[0] += N_vals[a] * cv[a][0];
                        x_phys[1] += N_vals[a] * cv[a][1];
                        x_phys[2] += N_vals[a] * cv[a][2];
                    }

                    // Jacobian matrix J_ij = dx_i / dξ_j (manual 3x3)
                    double J00 = 0, J01 = 0, J02 = 0;
                    double J10 = 0, J11 = 0, J12 = 0;
                    double J20 = 0, J21 = 0, J22 = 0;
                    for (int a = 0; a < 8; ++a) {
                        J00 += dN[a][0] * cv[a][0];
                        J01 += dN[a][1] * cv[a][0];
                        J02 += dN[a][2] * cv[a][0];
                        J10 += dN[a][0] * cv[a][1];
                        J11 += dN[a][1] * cv[a][1];
                        J12 += dN[a][2] * cv[a][1];
                        J20 += dN[a][0] * cv[a][2];
                        J21 += dN[a][1] * cv[a][2];
                        J22 += dN[a][2] * cv[a][2];
                    }

                    // det(J) = J00*J11*J22 + J01*J12*J20 + J02*J10*J21
                    //        - J00*J12*J21 - J01*J10*J22 - J02*J11*J20
                    double detJ = J00 * (J11 * J22 - J12 * J21) + J01 * (J12 * J20 - J10 * J22) +
                                  J02 * (J10 * J21 - J11 * J20);

                    // Inverse via cofactor matrix / det
                    // J^{-1}_ij = cofactor(J)_{ji} / det(J)
                    double Jinv00, Jinv01, Jinv02;
                    double Jinv10, Jinv11, Jinv12;
                    double Jinv20, Jinv21, Jinv22;
                    if (detJ > 0) {
                        double inv_det = 1.0 / detJ;
                        Jinv00 = (J11 * J22 - J12 * J21) * inv_det;
                        Jinv01 = (J02 * J21 - J01 * J22) * inv_det;
                        Jinv02 = (J01 * J12 - J02 * J11) * inv_det;
                        Jinv10 = (J12 * J20 - J10 * J22) * inv_det;
                        Jinv11 = (J00 * J22 - J02 * J20) * inv_det;
                        Jinv12 = (J02 * J10 - J00 * J12) * inv_det;
                        Jinv20 = (J10 * J21 - J11 * J20) * inv_det;
                        Jinv21 = (J01 * J20 - J00 * J21) * inv_det;
                        Jinv22 = (J00 * J11 - J01 * J10) * inv_det;
                    } else {
                        Jinv00 = 0;
                        Jinv01 = 0;
                        Jinv02 = 0;
                        Jinv10 = 0;
                        Jinv11 = 0;
                        Jinv12 = 0;
                        Jinv20 = 0;
                        Jinv21 = 0;
                        Jinv22 = 0;
                    }

                    // Store results
                    int64_t base = ((e * ngll + i) * ngll + j) * ngll + k;

                    int64_t coord_off = base * 3;
                    res.jacobian[base] = detJ;

                    int64_t dxi_off = base * 9;
                    res.dxi_dx[dxi_off + 0] = Jinv00;
                    res.dxi_dx[dxi_off + 1] = Jinv01;
                    res.dxi_dx[dxi_off + 2] = Jinv02;
                    res.dxi_dx[dxi_off + 3] = Jinv10;
                    res.dxi_dx[dxi_off + 4] = Jinv11;
                    res.dxi_dx[dxi_off + 5] = Jinv12;
                    res.dxi_dx[dxi_off + 6] = Jinv20;
                    res.dxi_dx[dxi_off + 7] = Jinv21;
                    res.dxi_dx[dxi_off + 8] = Jinv22;

                    res.mass[base] = detJ * w3[(i * ngll + j) * ngll + k];

                    // CFL: store coords for second pass computation
                    res.coords[coord_off + 0] = x_phys[0];
                    res.coords[coord_off + 1] = x_phys[1];
                    res.coords[coord_off + 2] = x_phys[2];

                    // PML damping ramp (compute for all nodes, Python masks with is_pml)
                    double damp_val = 0.0;
                    double x = x_phys[0], y = x_phys[1], z = x_phys[2];
                    // Face 0: xmin
                    if (pml_width[0] > 0 && x < pml_start[0]) {
                        double r = (pml_start[0] - x) / pml_width[0];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    // Face 1: xmax
                    if (pml_width[1] > 0 && x > pml_start[1]) {
                        double r = (x - pml_start[1]) / pml_width[1];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    // Face 2: ymin
                    if (pml_width[2] > 0 && y < pml_start[2]) {
                        double r = (pml_start[2] - y) / pml_width[2];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    // Face 3: ymax
                    if (pml_width[3] > 0 && y > pml_start[3]) {
                        double r = (y - pml_start[3]) / pml_width[3];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    // Face 4: zmin
                    if (pml_width[4] > 0 && z < pml_start[4]) {
                        double r = (pml_start[4] - z) / pml_width[4];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    // Face 5: zmax
                    if (pml_width[5] > 0 && z > pml_start[5]) {
                        double r = (z - pml_start[5]) / pml_width[5];
                        if (r > damp_val)
                            damp_val = r;
                    }
                    res.damping[base] = std::min(damp_val, 1.0);
                }
            }
        }
    }

    // Second pass: compute CFL min spacing from stored coords (single-threaded)
    for (int64_t e = 0; e < n_cell; ++e) {
        for (int i = 0; i < ngll; ++i) {
            for (int j = 0; j < ngll; ++j) {
                for (int k = 0; k < ngll; ++k) {
                    int64_t base = ((e * ngll + i) * ngll + j) * ngll + k;
                    int64_t off = base * 3;
                    double x = res.coords[off + 0];
                    double y = res.coords[off + 1];
                    double z = res.coords[off + 2];
                    if (i + 1 < ngll) {
                        int64_t noff = (((e * ngll + (i + 1)) * ngll + j) * ngll + k) * 3;
                        double dx = x - res.coords[noff + 0];
                        double dy = y - res.coords[noff + 1];
                        double dz = z - res.coords[noff + 2];
                        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist < h_min)
                            h_min = dist;
                    }
                    if (j + 1 < ngll) {
                        int64_t noff = (((e * ngll + i) * ngll + (j + 1)) * ngll + k) * 3;
                        double dx = x - res.coords[noff + 0];
                        double dy = y - res.coords[noff + 1];
                        double dz = z - res.coords[noff + 2];
                        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist < h_min)
                            h_min = dist;
                    }
                    if (k + 1 < ngll) {
                        int64_t noff = (((e * ngll + i) * ngll + j) * ngll + (k + 1)) * 3;
                        double dx = x - res.coords[noff + 0];
                        double dy = y - res.coords[noff + 1];
                        double dz = z - res.coords[noff + 2];
                        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist < h_min)
                            h_min = dist;
                    }
                }
            }
        }
    }

    // CFL: h_min only. Python combines with vp_max: cfl_dt = cfl_safety * h_min / vp_max
    res.cfl_dt = (h_min < 1e29) ? h_min : 0.0;  // just h_min, not full dt

    return res;
}

// -----------------------------------------------------------------------
// Write results to mesh.h5
// -----------------------------------------------------------------------
static void write_results(const char* mesh_path, int64_t n_cell, int ngll,
                          const ComputeResult& res, double cfl_safety, double h_min) {
    hid_t fid = open_or_fail(mesh_path, H5F_ACC_RDWR);

    // Ensure /field/element/ group exists
    hid_t fld_gid;
    if (H5Lexists(fid, "field", H5P_DEFAULT)) {
        fld_gid = H5Gopen2(fid, "field", H5P_DEFAULT);
    } else {
        fld_gid = H5Gcreate2(fid, "field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    hid_t elem_gid;
    if (H5Lexists(fld_gid, "element", H5P_DEFAULT)) {
        elem_gid = H5Gopen2(fld_gid, "element", H5P_DEFAULT);
    } else {
        elem_gid = H5Gcreate2(fld_gid, "element", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }

    hsize_t nc = static_cast<hsize_t>(n_cell);
    hsize_t ng = static_cast<hsize_t>(ngll);

    // Delete existing datasets before recreating (in case of re-run)
    const char* dset_names[] = {"coords", "dxi_dx", "jacobian", "mass", "damping"};
    for (auto dn : dset_names) {
        if (H5Lexists(elem_gid, dn, H5P_DEFAULT))
            H5Ldelete(elem_gid, dn, H5P_DEFAULT);
    }
    write_5d_double(elem_gid, "coords", nc, ng, 3, res.coords.data());
    write_5d_double(elem_gid, "dxi_dx", nc, ng, 9, res.dxi_dx.data());
    write_4d_double(elem_gid, "jacobian", nc, ng, res.jacobian.data());
    write_4d_double(elem_gid, "mass", nc, ng, res.mass.data());
    write_4d_double(elem_gid, "damping", nc, ng, res.damping.data());

    H5Gclose(elem_gid);
    H5Gclose(fld_gid);

    // Write info group with solver metadata
    hid_t info_gid;
    if (H5Lexists(fid, "info", H5P_DEFAULT)) {
        info_gid = H5Gopen2(fid, "info", H5P_DEFAULT);
    } else {
        info_gid = H5Gcreate2(fid, "info", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    write_scalar_attr(info_gid, "h_min", h_min);
    write_scalar_attr(info_gid, "cfl_safety", cfl_safety);
    write_scalar_attr(info_gid, "cfl_dt", res.cfl_dt);

    H5Gclose(info_gid);
    H5Fclose(fid);
}

// -----------------------------------------------------------------------
// Print CFL info for Python to capture
// -----------------------------------------------------------------------
static void print_cfl_info(double h_min, double cfl_safety) {
    // Print in a format Python can parse
    printf("H_MIN=%.15e\n", h_min);
    printf("CFL_SAFETY=%.15e\n", cfl_safety);
    fflush(stdout);
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: gf_preprocess_cpp <mesh.h5> <N> <cfl_safety>\n"
                "       [pml_xmin pml_xmax pml_ymin pml_ymax pml_zmin pml_zmax]\n");
        return 1;
    }

    const char* mesh_path = argv[1];
    int N = std::atoi(argv[2]);
    double cfl_safety = std::atof(argv[3]);

    double pml_thickness[6] = {0, 0, 0, 0, 0, 0};
    if (argc >= 10) {
        for (int f = 0; f < 6; ++f)
            pml_thickness[f] = std::atof(argv[4 + f]);
    }

    if (N < 1) {
        fprintf(stderr, "ERROR: N must be >= 1, got %d\n", N);
        return 1;
    }
    if (cfl_safety <= 0 || cfl_safety >= 1) {
        fprintf(stderr, "ERROR: cfl_safety must be in (0,1), got %g\n", cfl_safety);
        return 1;
    }

    // Read topology
    Topology topo = read_topology(mesh_path);
    fprintf(stderr, "Topology: n_cell=%lld, n_vertex=%lld, n_surface=%lld, n_edge=%lld\n",
            (long long)topo.n_cell, (long long)topo.n_vertex, (long long)topo.n_surface,
            (long long)topo.n_edge);

    // Read domain bounds from /domain/ attrs (written by mesh generator or Python)
    double domain_bounds[6] = {0, 0, 0, 0, 0, 0};
    {
        hid_t fid = open_or_fail(mesh_path, H5F_ACC_RDONLY);
        hid_t dom_gid = H5Gopen2(fid, "domain", H5P_DEFAULT);
        if (dom_gid >= 0) {
            read_attr_double(dom_gid, "xmin", domain_bounds[0]);
            read_attr_double(dom_gid, "xmax", domain_bounds[1]);
            read_attr_double(dom_gid, "ymin", domain_bounds[2]);
            read_attr_double(dom_gid, "ymax", domain_bounds[3]);
            read_attr_double(dom_gid, "zmin", domain_bounds[4]);
            read_attr_double(dom_gid, "zmax", domain_bounds[5]);
            H5Gclose(dom_gid);
        } else {
            // Fallback: infer from vertex coords
            fprintf(stderr, "WARNING: /domain/ not found, inferring from vertex coords\n");
            const double* vc = topo.vertex_to_coord.data();
            int64_t nv = topo.n_vertex;
            domain_bounds[0] = domain_bounds[2] = domain_bounds[4] = 1e30;
            domain_bounds[1] = domain_bounds[3] = domain_bounds[5] = -1e30;
            for (int64_t i = 0; i < nv; ++i) {
                if (vc[i * 3 + 0] < domain_bounds[0])
                    domain_bounds[0] = vc[i * 3 + 0];
                if (vc[i * 3 + 0] > domain_bounds[1])
                    domain_bounds[1] = vc[i * 3 + 0];
                if (vc[i * 3 + 1] < domain_bounds[2])
                    domain_bounds[2] = vc[i * 3 + 1];
                if (vc[i * 3 + 1] > domain_bounds[3])
                    domain_bounds[3] = vc[i * 3 + 1];
                if (vc[i * 3 + 2] < domain_bounds[4])
                    domain_bounds[4] = vc[i * 3 + 2];
                if (vc[i * 3 + 2] > domain_bounds[5])
                    domain_bounds[5] = vc[i * 3 + 2];
            }
        }
        H5Fclose(fid);
    }

    fprintf(stderr, "Domain: x=[%g,%g] y=[%g,%g] z=[%g,%g]\n", domain_bounds[0], domain_bounds[1],
            domain_bounds[2], domain_bounds[3], domain_bounds[4], domain_bounds[5]);
    fprintf(stderr, "N=%d, ngll=%d, cfl_safety=%g\n", N, N + 1, cfl_safety);
    fprintf(stderr, "PML thickness: [%g %g %g %g %g %g]\n", pml_thickness[0], pml_thickness[1],
            pml_thickness[2], pml_thickness[3], pml_thickness[4], pml_thickness[5]);

    // Compute
    ComputeResult res = compute_all(topo, N, cfl_safety, pml_thickness, domain_bounds);

    fprintf(stderr, "Computation done. h_min=%.15e\n", res.cfl_dt);

    // Write results to HDF5
    write_results(mesh_path, topo.n_cell, N + 1, res, cfl_safety, res.cfl_dt);

    // Print CFL info for Python
    print_cfl_info(res.cfl_dt, cfl_safety);

    fprintf(stderr, "Done. Wrote coords, dxi_dx, jacobian, mass, damping, info\n");
    return 0;
}