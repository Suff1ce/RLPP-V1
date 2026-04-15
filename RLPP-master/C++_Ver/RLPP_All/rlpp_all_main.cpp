// RLPP_All: umbrella launcher that dispatches to existing executables.
//
// Goal: single user-facing entrypoint without forcing a large refactor
// of Phase_1 / Phase_3 code into libraries.
//
// This executable spawns sibling binaries built elsewhere, passing through arguments.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void usage() {
    std::cout
        << "Usage: RLPP_All <command> [args...]\n"
        << "\n"
        << "Commands (dispatch targets):\n"
        << "  verify-all             run key parity/verifiers (expects Python-exported CSV cases)\n"
        << "  phase1                 -> RLPP_Phase_1.exe\n"
        << "  phase3                 -> RLPP_Phase_3.exe\n"
        << "  phase3-run             -> RLPP_Phase_3_Run.exe\n"
        << "  phase3-trainer         -> RLPP_Phase_3_Trainer.exe\n"
        << "  phase3-loader-verify   -> RLPP_Phase_3_LoaderEmuVerify.exe\n"
        << "  phase3-fullloop-manual -> RLPP_Phase_3_FullLoopVerify.exe\n"
        << "  phase3-fullloop-trained-> RLPP_Phase_3_FullLoopTrainedVerify.exe\n"
        << "  phase3-run-case-verify -> RLPP_Phase_3_RunCaseVerify.exe\n"
        << "  hw-udp-loopback        -> RLPP_hardware_udp_loopback.exe\n"
        << "  realtime-udp-infer     -> RLPP_realtime_udp_infer.exe\n"
        << "\n"
        << "Notes:\n"
        << "  - Build the target executables first (Phase_1/Phase_3).\n"
        << "  - RLPP_All locates them relative to its own path.\n";
}

static fs::path exe_dir(char** argv) {
    // argv[0] is enough for our use in this repo (no need for platform APIs).
    return fs::absolute(fs::path(argv[0])).parent_path();
}

static fs::path pick_existing(const std::vector<fs::path>& candidates) {
    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(p, ec)) return p;
    }
    return {};
}

static std::string quote_arg(const std::string& a) {
    // Minimal quoting for Windows cmd / PowerShell compatibility:
    // wrap in quotes if spaces; escape embedded quotes.
    bool need = false;
    for (char c : a) if (c == ' ' || c == '\t') need = true;
    if (!need) return a;
    std::string out = "\"";
    for (char c : a) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static int spawn_and_wait(const fs::path& exe, const std::vector<std::string>& args) {
    std::string cmd = quote_arg(exe.string());
    for (const auto& a : args) {
        cmd += " ";
        cmd += quote_arg(a);
    }
    return std::system(cmd.c_str());
}

static bool is_dir(const fs::path& p) {
    std::error_code ec;
    return fs::is_directory(p, ec);
}

static std::optional<fs::path> require_exe(const fs::path& p, const std::string& label) {
    std::error_code ec;
    if (fs::is_regular_file(p, ec)) return p;
    std::cerr << "Missing executable: " << label << " at " << p.string() << "\n";
    return std::nullopt;
}

static int verify_all(const fs::path& cpp_ver) {
    // We intentionally do NOT auto-run Python exporters here.
    // This wrapper only runs C++ verifiers if the expected case dirs exist.
    const fs::path phase3_release = cpp_ver / "Phase_3" / "build" / "Release";

    const fs::path exe_phase3 = phase3_release / "RLPP_Phase_3.exe";
    const fs::path exe_loader = phase3_release / "RLPP_Phase_3_LoaderEmuVerify.exe";
    const fs::path exe_full_manual = phase3_release / "RLPP_Phase_3_FullLoopVerify.exe";
    const fs::path exe_full_trained = phase3_release / "RLPP_Phase_3_FullLoopTrainedVerify.exe";
    const fs::path exe_run_case = phase3_release / "RLPP_Phase_3_RunCaseVerify.exe";

    // Exported case directories (from Python exporters).
    const fs::path testdata = cpp_ver / "Phase_3" / "testdata";
    const fs::path loader_case = cpp_ver / "Phase_3" / "loader_emulator_case";
    const fs::path full_manual_case = cpp_ver / "Phase_3" / "full_loop_emulator_case";
    const fs::path full_d01_case = cpp_ver / "Phase_3" / "full_loop_decoding01_case";
    const fs::path full_d02_case = cpp_ver / "Phase_3" / "full_loop_decoding02_case";
    const fs::path run_case_d01 = cpp_ver / "Phase_3" / "run_case_decoding01";
    const fs::path run_case_d02 = cpp_ver / "Phase_3" / "run_case_decoding02";
    const fs::path run_case_manual = cpp_ver / "Phase_3" / "run_case_manual";

    // ---- Ensure required executables exist ----
    if (!require_exe(exe_phase3, "RLPP_Phase_3")) return 1;
    if (!require_exe(exe_loader, "RLPP_Phase_3_LoaderEmuVerify")) return 1;
    if (!require_exe(exe_full_manual, "RLPP_Phase_3_FullLoopVerify")) return 1;
    if (!require_exe(exe_full_trained, "RLPP_Phase_3_FullLoopTrainedVerify")) return 1;
    if (!require_exe(exe_run_case, "RLPP_Phase_3_RunCaseVerify")) return 1;

    // ---- Ensure required case dirs exist ----
    bool ok = true;
    auto need_dir = [&](const fs::path& p, const char* how) {
        if (!is_dir(p)) {
            std::cerr << "Missing case directory: " << p.string() << "\n"
                      << "  Generate it by running: " << how << "\n";
            ok = false;
        }
    };

    need_dir(testdata, "python Python_Ver/export_phase3_reference.py");
    need_dir(loader_case, "python Python_Ver/export_loader_emulator_reference.py");
    need_dir(full_manual_case, "python Python_Ver/export_phase3_full_loop_emulator_case.py");
    need_dir(full_d01_case, "python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 01");
    need_dir(full_d02_case, "python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 02");
    need_dir(run_case_d01, "python Python_Ver/export_phase3_run_case.py --model 01");
    need_dir(run_case_d02, "python Python_Ver/export_phase3_run_case.py --model 02");
    need_dir(run_case_manual, "python Python_Ver/export_phase3_run_case.py --model manual");

    if (!ok) {
        std::cerr << "verify-all aborted: missing required exported cases.\n";
        return 2;
    }

    // ---- Run verifiers (fail-fast) ----
    auto run = [&](const fs::path& exe, const std::vector<std::string>& args, const char* label) -> int {
        std::cout << "\n== " << label << " ==\n";
        const int rc = spawn_and_wait(exe, args);
        if (rc != 0) {
            std::cerr << label << " FAILED with exit code " << rc << "\n";
        }
        return rc;
    };

    // RL math parity
    if (int rc = run(exe_phase3, {"--testdata", testdata.string()}, "Phase3 RL math parity"); rc != 0) return rc;

    // DataLoader + emulator parity
    if (int rc = run(exe_loader, {}, "Loader + emulator parity"); rc != 0) return rc;

    // Full loop manual
    if (int rc = run(exe_full_manual, {full_manual_case.string()}, "Full-loop manual decoder parity"); rc != 0) return rc;

    // Full loop trained (01 + 02)
    if (int rc = run(exe_full_trained, {full_d01_case.string()}, "Full-loop trained decoder01 parity"); rc != 0) return rc;
    if (int rc = run(exe_full_trained, {full_d02_case.string()}, "Full-loop trained decoder02 parity"); rc != 0) return rc;

    // Runner-style cases
    if (int rc = run(exe_run_case, {run_case_d01.string()}, "Runner-style decoding01 parity"); rc != 0) return rc;
    if (int rc = run(exe_run_case, {run_case_d02.string()}, "Runner-style decoding02 parity"); rc != 0) return rc;
    if (int rc = run(exe_run_case, {run_case_manual.string()}, "Runner-style manual parity"); rc != 0) return rc;

    std::cout << "\nVERIFY-ALL: PASS\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        usage();
        return 0;
    }

    const fs::path here = exe_dir(argv);
    // Expected layout when built from C++_Ver/RLPP_All/build/Release:
    //   .../C++_Ver/RLPP_All/build/Release/RLPP_All.exe
    // So C++_Ver is 3 levels up.
    const fs::path cpp_ver = here / ".." / ".." / "..";

    if (cmd == "verify-all") {
        return verify_all(cpp_ver);
    }

    auto phase1 = [&](const char* name) {
        return pick_existing({
            cpp_ver / "Phase_1" / "build" / "Release" / name,
            cpp_ver / "Phase_1" / "build" / name,
        });
    };
    auto phase3 = [&](const char* name) {
        return pick_existing({
            cpp_ver / "Phase_3" / "build" / "Release" / name,
            cpp_ver / "Phase_3" / "build" / name,
        });
    };

    fs::path target;
    if (cmd == "phase1") target = phase1("RLPP_Phase_1.exe");
    else if (cmd == "hw-udp-loopback") target = phase1("RLPP_hardware_udp_loopback.exe");
    else if (cmd == "realtime-udp-infer") target = phase1("RLPP_realtime_udp_infer.exe");
    else if (cmd == "phase3") target = phase3("RLPP_Phase_3.exe");
    else if (cmd == "phase3-run") target = phase3("RLPP_Phase_3_Run.exe");
    else if (cmd == "phase3-trainer") target = phase3("RLPP_Phase_3_Trainer.exe");
    else if (cmd == "phase3-loader-verify") target = phase3("RLPP_Phase_3_LoaderEmuVerify.exe");
    else if (cmd == "phase3-fullloop-manual") target = phase3("RLPP_Phase_3_FullLoopVerify.exe");
    else if (cmd == "phase3-fullloop-trained") target = phase3("RLPP_Phase_3_FullLoopTrainedVerify.exe");
    else if (cmd == "phase3-run-case-verify") target = phase3("RLPP_Phase_3_RunCaseVerify.exe");
    else {
        std::cerr << "Unknown command: " << cmd << "\n\n";
        usage();
        return 1;
    }

    if (target.empty()) {
        std::cerr << "Could not find target executable for command: " << cmd << "\n"
                  << "Build Phase_1/Phase_3 first, then re-run.\n";
        return 1;
    }

    std::vector<std::string> pass;
    for (int i = 2; i < argc; ++i) pass.push_back(argv[i]);
    return spawn_and_wait(target, pass);
}

