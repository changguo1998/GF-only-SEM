// forward/include/gf/logger.hpp
//
// Simple timestamped file logger for forward solver.
// Writes to log/forward_{direction}_{rank}.log, also echoes to stdout (rank 0).

#pragma once

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace gf {

class Logger {
public:
    Logger(const std::string& direction, int rank) : rank_(rank) {
        // Create log directory
        mkdir("log", 0755);

        std::ostringstream filename;
        filename << "log/forward_" << direction << "_" << rank << ".log";
        file_.open(filename.str(), std::ios::out | std::ios::trunc);
        if (!file_.is_open() && rank_ == 0) {
            std::cerr << "[Rank 0] WARNING: Cannot open log file: " << filename.str() << std::endl;
        }
    }

    ~Logger() { close(); }

    void info(const std::string& msg) {
        write("INFO", msg);
        if (rank_ == 0) {
            std::cout << "gf_solver: " << msg << std::endl;
        }
    }

    void debug(const std::string& msg) { write("DEBUG", msg); }

    void error(const std::string& msg) {
        write("ERROR", msg);
        if (rank_ == 0) {
            std::cerr << "[Rank 0] " << msg << std::endl;
        }
    }

    void raw(const std::string& msg) {
        // Write to file without timestamp prefix; echo to stdout for rank 0
        if (file_.is_open()) {
            file_ << msg << std::endl;
        }
        if (rank_ == 0) {
            std::cout << msg << std::endl;
        }
    }

    void progress(const std::string& msg) {
        // Log file: full timestamped line (no in-place).
        // No carriage-return or ANSI escape sequences here — just clean lines.
        if (file_.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) %
                1000;
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
            file_ << time_buf << "." << std::setw(3) << std::setfill('0') << ms.count()
                  << " [PROG] " << msg << std::endl;
        }
        // Terminal: in-place update directly to the controlling terminal.
        //
        // MPI I/O forwarding (orte/iof) reads rank output from pipe/socket pairs
        // in a line-buffered fashion — \r without \n is never flushed to the
        // terminal.  Switching between stdout and stderr doesn't help; MPI's IOF
        // layer wraps both.
        //
        // Writing to /dev/tty bypasses MPI's output pipes entirely.  Data goes
        // straight to the terminal device, so \r and ANSI escape sequences work
        // as expected.  If /dev/tty is unavailable (batch job, CI, no terminal),
        // progress is still captured in the log file.
        if (rank_ == 0) {
            FILE* tty = fopen("/dev/tty", "w");
            if (tty) {
                fwrite("\r\x1b[K", 1, 4, tty);
                fwrite(msg.data(), 1, msg.size(), tty);
                fflush(tty);
                fclose(tty);
            }
        }
    }

    void progress_done() {
        // Write newline to /dev/tty so subsequent output starts on a fresh line.
        if (rank_ == 0) {
            FILE* tty = fopen("/dev/tty", "w");
            if (tty) {
                fputc('\n', tty);
                fclose(tty);
            }
        }
    }

    void close() {
        if (file_.is_open()) {
            file_.close();
        }
    }

private:
    std::ofstream file_;
    int rank_;

    void write(const std::string& level, const std::string& msg) {
        if (!file_.is_open())
            return;
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

        file_ << time_buf << "." << std::setw(3) << std::setfill('0') << ms.count() << " ["
              << std::setw(5) << std::setfill(' ') << level << "] " << msg << std::endl;
    }
};

}  // namespace gf