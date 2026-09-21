#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "gf/restart.hpp"
#include "gf/types.hpp"

using namespace gf;

TEST_CASE("Restart round-trip preserves C-PML state", "[restart][cpml]") {
    const std::filesystem::path output_dir = "test_restart_cpml_output";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directory(output_dir);

    std::vector<double> displacement = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> velocity = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<double> acceleration = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
    std::vector<double> damping = {0.01, 0.02};

    RankData part;
    part.has_cpml = true;
    part.pml_displ_old = {1.1, 1.2, 1.3};
    part.pml_displ_new = {2.1, 2.2, 2.3};
    part.rmemory_displ = {3.1, 3.2, 3.3, 3.4};
    part.rmemory_strain = {4.1, 4.2, 4.3, 4.4, 4.5};

    {
        RestartWriter writer(output_dir.string(), "x", 0, 1, 2, true, 2);
        writer.write(20, 0.2, displacement, velocity, acceleration, damping, &part);
        writer.close();
    }

    RestartState state = read_restart(output_dir.string(), "x", 0);
    REQUIRE(state.has_cpml);
    REQUIRE(state.pml_displ_old == part.pml_displ_old);
    REQUIRE(state.pml_displ_new == part.pml_displ_new);
    REQUIRE(state.rmemory_displ == part.rmemory_displ);
    REQUIRE(state.rmemory_strain == part.rmemory_strain);

    const std::filesystem::path restart_path = output_dir / "x/restart_0.h5";
    hid_t file = H5Fopen(restart_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    REQUIRE(file >= 0);
    REQUIRE(H5Ldelete(file, "rmemory_strain", H5P_DEFAULT) >= 0);
    H5Fclose(file);
    REQUIRE_THROWS_AS(read_restart(output_dir.string(), "x", 0), std::runtime_error);

    std::filesystem::remove_all(output_dir);
}
