#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/flat_matrix.hpp"
#include "../include/activation_softmax.hpp"   // deine Softmax-Klasse

// ---------- kleine Hilfsroutinen ----------
constexpr double EPS_FWD = 1e-6;
constexpr double EPS_BWD = 1e-6;
constexpr double EPS_SUM = 1e-12;

bool approx(double a, double b, double eps) { return std::fabs(a - b) <= eps; }

void expect_throw(const std::function<void()>& fn, const char* msg) {
    bool ok = false;
    try { fn(); }
    catch (const std::invalid_argument&) { ok = true; }
    catch (...) {}
    if (!ok) {
        std::cerr << "FAILED (no exception): " << msg << "\n";
        std::abort();
    }
}

FlatMatrix from2D(const std::vector<std::vector<double>>& v) {
    int R = static_cast<int>(v.size());
    int C = static_cast<int>(v.empty() ? 0 : v[0].size());
    FlatMatrix M(R, C, 0.0);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            M.set(i, j, v[i][j]);
    return M;
}

double max_abs_diff(const FlatMatrix& A,
                    const std::vector<std::vector<double>>& ref) {
    int R = A.rows(), C = A.cols();
    double m = 0.0;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            m = std::max(m, std::fabs(A.get(i, j) - ref[i][j]));
    return m;
}

double row_sum(const FlatMatrix& M, int i) {
    double s = 0.0;
    for (int j = 0; j < M.cols(); ++j) s += M.get(i, j);
    return s;
}

void print_matrix(const char* name, const FlatMatrix& M) {
    std::cout << name << " (" << M.rows() << "x" << M.cols() << "):\n";
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            std::cout << M.get(i, j);
            if (j + 1 < M.cols()) std::cout << ' ';
        }
        std::cout << '\n';
    }
}

// ---------- Tests ----------
void test_sanity_1x3_backward() {
    // Wir testen nur die Backward-Formel mit einem bekannten Beispiel:
    // s = [0.8, 0.1, 0.1], g = [1, 0, -1] -> dz = [0.24, -0.07, -0.17]
    ActivationSoftmax sm;
    // Trick: Wir setzen direkt sm.output, um nur die Backward-Formel zu verifizieren.
    sm.output = from2D({{0.8, 0.1, 0.1}});

    FlatMatrix dvalues = from2D({{1.0, 0.0, -1.0}});
    sm.backward(dvalues);

    std::vector<std::vector<double>> ref = {{0.24, -0.07, -0.17}};

    // Form
    assert(sm.dinputs.rows() == 1 && sm.dinputs.cols() == 3);

    // Werte
    double mad = max_abs_diff(sm.dinputs, ref);
    if (mad > EPS_BWD) {
        print_matrix("got dinputs", sm.dinputs);
        std::cerr << "Sanity 1x3 backward: max abs diff = " << mad << " > " << EPS_BWD << "\n";
        std::abort();
    }

    // Zeilensumme ≈ 0
    if (std::fabs(row_sum(sm.dinputs, 0)) > EPS_SUM) {
        std::cerr << "Sanity 1x3 backward: row sum not ~0\n";
        std::abort();
    }

    std::cout << "Sanity 1x3 backward ✔\n";
}

void test_2x3_forward_and_backward() {
    // Forward: bekannte Inputs & Sollwerte
    auto X = from2D({
        {2.0, 1.0, 0.5},
        {1.0, 3.0, 2.0}
    });
    std::vector<std::vector<double>> ref_softmax = {
        {0.62853172, 0.23122390, 0.14024438},
        {0.09003057, 0.66524096, 0.24472847}
    };

    ActivationSoftmax sm;
    sm.forward(X);

    // Forward-Check
    double mad_fwd = max_abs_diff(sm.output, ref_softmax);
    if (mad_fwd > EPS_FWD) {
        print_matrix("got softmax", sm.output);
        std::cerr << "Forward 2x3: max abs diff = " << mad_fwd << " > " << EPS_FWD << "\n";
        std::abort();
    }
    // Zeilensummen = 1
    for (int i = 0; i < sm.output.rows(); ++i) {
        if (!approx(row_sum(sm.output, i), 1.0, EPS_SUM)) {
            std::cerr << "Forward 2x3: row " << i << " does not sum to 1\n";
            std::abort();
        }
    }

    // Backward: Upstream-Gradient & Sollwerte
    auto dvalues = from2D({
        { 1.0,  0.0, -1.0},
        { 0.5,  0.5, -1.0}
    });
    std::vector<std::vector<double>> ref_dinputs = {
        { 0.32162764, -0.11290370, -0.20872394},
        { 0.03304957,  0.24420510, -0.27725467}
    };

    sm.backward(dvalues);

    double mad_bwd = max_abs_diff(sm.dinputs, ref_dinputs);
    if (mad_bwd > EPS_BWD) {
        print_matrix("got dinputs", sm.dinputs);
        std::cerr << "Backward 2x3: max abs diff = " << mad_bwd << " > " << EPS_BWD << "\n";
        std::abort();
    }
    // Zeilensummen ≈ 0
    for (int i = 0; i < sm.dinputs.rows(); ++i) {
        if (std::fabs(row_sum(sm.dinputs, i)) > EPS_SUM) {
            std::cerr << "Backward 2x3: row " << i << " sum not ~0\n";
            std::abort();
        }
    }

    std::cout << "Forward+Backward 2x3 ✔\n";
}

void test_error_cases() {
    // Fehlerfälle: falsche Rows/Cols sollen invalid_argument werfen
    ActivationSoftmax sm;
    auto X = from2D({{2.0,1.0,0.5},{1.0,3.0,2.0}});
    sm.forward(X);

    // rows mismatch
    auto bad_rows = from2D({{1,0,-1},{0.5,0.5,-1},{7,7,7}});
    expect_throw([&](){ sm.backward(bad_rows); },
        "rows mismatch not detected");

    // cols mismatch
    auto bad_cols = from2D({{1,0},{0.5,0.5}});
    expect_throw([&](){ sm.backward(bad_cols); },
        "cols mismatch not detected");

    std::cout << "Error cases ✔\n";
}

void test_smoke_large() {
    // Großer Smoke-Test auf Stabilität (kein genauer Sollwert)
    const int R = 500, C = 10;
    FlatMatrix X(R, C, 0.0);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            X.set(i, j, (i*7 + j*11) % 13 - 6);  // bisschen Variation

    ActivationSoftmax sm;
    sm.forward(X);

    FlatMatrix d(R, C, 0.0);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            d.set(i, j, ((i+j)%3==0) ? 1.0 : -0.5);

    sm.backward(d);

    // Zeilensummen ~ 0
    for (int i = 0; i < R; ++i) {
        if (std::fabs(row_sum(sm.dinputs, i)) > 1e-9) {
            std::cerr << "Smoke: row " << i << " sum not ~0\n";
            std::abort();
        }
    }
    std::cout << "Smoke large ✔\n";
}

// ---------- main ----------
int main() {
    test_sanity_1x3_backward();
    test_2x3_forward_and_backward();
    test_error_cases();
    test_smoke_large();

    std::cout << "All Softmax backward checks passed ✅\n";
    return 0;
}
