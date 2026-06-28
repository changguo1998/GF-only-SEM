// forward/include/gf/logger.hpp
//
// Simple timestamped file logger for forward solver.
// Writes to log/forward_{direction}_{rank}.log, also echoes to stdout (rank 0).

#pragma once

#include <sys/stat.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace gf {

class Logger {
public:
    Logger(const std::string& direction, int rank) : rank_(rank), first_progress_(true) {
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
        // In-place progress line.
        // Writes timestamped to log file. On stdout:
        //   first call:  \r\e[K msg \n  (clear current line, write, newline flush)
        //   later calls: \e[A\r\e[K msg \n  (cursor up, clear, write)
        // \n ensures MPI line-buffered stdout flushes each update.
        // \e[A rewinds to keep the progress line visually in-place.
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
        if (rank_ == 0) {
            if (first_progress_) {
                std::cout << "\r\x1b[K" << msg << std::endl;
                first_progress_ = false;
            } else {
                std::cout << "\x1b[A\r\x1b[K" << msg << std::endl;
            }
        }
    }

    void progress_done() {
        // Already on a fresh line after the last progress's \n — nothing needed.
        // Reset state for potential reuse.
        if (rank_ == 0) {
            first_progress_ = true;
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
    bool first_progress_;

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