/* gf_preprocess.cpp — unified preprocess entry point
 *
 * Dispatches to stage1 (GLL geometry, PML) or stage2 (λ/μ, solver_dt, stats).
 *
 * Usage:
 *   gf_preprocess stage1 <model.h5> [--N N] [--cfl-safety val] ...
 *   gf_preprocess stage2 <model.h5>
 *
 * Both stages are self-describing — run without arguments for usage.
 */

#include <cstdio>
#include <cstring>

extern int stage1_main(int argc, char** argv);
extern int stage2_main(int argc, char** argv);

static void usage() {
    fprintf(stderr,
            "Usage: gf_preprocess <stage1|stage2> [args...]\n"
            "\n"
            "  stage1  Compute GLL geometry, PML damping, CFL h_min.\n"
            "          gf_preprocess stage1 --help\n"
            "  stage2  Compute λ/μ, solver_dt, pre-flight statistics.\n"
            "          gf_preprocess stage2 <model.h5>\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char* subcommand = argv[1];

    if (std::strcmp(subcommand, "stage1") == 0) {
        return stage1_main(argc - 1, argv + 1);
    }
    if (std::strcmp(subcommand, "stage2") == 0) {
        return stage2_main(argc - 1, argv + 1);
    }

    // Check for --help or -h before unknown subcommand
    if (std::strcmp(subcommand, "--help") == 0 || std::strcmp(subcommand, "-h") == 0) {
        usage();
        return 0;
    }
    if (std::strcmp(subcommand, "--version") == 0 || std::strcmp(subcommand, "-V") == 0) {
        fprintf(stdout, "gf_preprocess v1.0.0\n");
        return 0;
    }

    fprintf(stderr, "ERROR: unknown subcommand: %s\n", subcommand);
    usage();
    return 1;
}