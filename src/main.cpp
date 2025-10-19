#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/flat_matrix.hpp"
#include "../include/categorical_cross_entropy.hpp"  // dein Header

// ---------- kleine Hilfen ----------
constexpr double EPS = 1e-8;

FlatMatrix from2D(const std::vector<std::vector<double>>& v) {
    int R = static_cast<int>(v.size());
    int C = static_cast<int>(v.empty() ? 0 : v[0].size());
    FlatMatrix M(R, C, 0.0);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            M.set(i, j, v[i][j]);
    return M;
}

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

bool approx(double a, double b, double eps = EPS) {
    return std::fabs(a - b) <= eps;
}

// ---------- Tests ----------
void test_single_sample_values() {
    LossCategoricalCrossEntropy loss;

    auto clip_prob = [](double p) {
        const double CLIP = 1e-7;
        if (p < CLIP) return CLIP;
        if (p > 1.0 - CLIP) return 1.0 - CLIP;
        return p;
    };

    // p_t = 1.0 -> wegen symmetrischem Clipping nicht exakt 0,
    // sondern -log(1 - 1e-7) ~ 1.00000005e-7
    {
        auto p = from2D({{1.0, 0.0, 0.0}});
        std::vector<int> y = {0};
        double L = loss.forward(p, y);

        const double expected = -std::log(clip_prob(1.0));
        // etwas großzügigere Toleranz für diesen Spezialfall
        if (std::fabs(L - expected) > 1e-9) {
            std::cerr << "Expected ~" << expected << " but got " << L << "\n";
            std::abort();
        }
        // zusätzlich: L ist endlich und >= 0
        assert(std::isfinite(L) && L >= 0.0);
    }

    // p_t = 0.5 -> ~0.69314718
    {
        auto p = from2D({{0.5, 0.5, 0.0}});
        std::vector<int> y = {0};
        double L = loss.forward(p, y);
        assert(approx(L, 0.6931471805599453, 1e-12));
    }

    // p_t = 0.1 -> ~2.30258509
    {
        auto p = from2D({{0.1, 0.9, 0.0}});
        std::vector<int> y = {0};
        double L = loss.forward(p, y);
        assert(approx(L, 2.302585092994046, 1e-12));
    }

    std::cout << "Single-sample CCE values ✔\n";
}

void test_two_sample_batch_label_and_onehot() {
    LossCategoricalCrossEntropy loss;

    // 2×3 Beispiel aus der Theorie
    auto p = from2D({
        {0.7, 0.2, 0.1},
        {0.1, 0.5, 0.4}
    });
    std::vector<int> y_labels = {0, 2};
    auto y_onehot = from2D({
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0}
    });

    // Erwarteter Mittelwert: ~0.63648283
    double L_labels = loss.forward(p, y_labels);
    double L_onehot = loss.forward(p, y_onehot);

    if (!approx(L_labels, 0.63648283) || !approx(L_onehot, 0.63648283)) {
        std::cerr << "Got L_labels=" << L_labels
                  << " L_onehot=" << L_onehot
                  << " expected ~0.63648283\n";
        std::abort();
    }
    // Gleichheit Label vs. One-Hot
    assert(approx(L_labels, L_onehot));

    std::cout << "Batch (labels & one-hot) ✔\n";
}

void test_clipping_edges() {
    LossCategoricalCrossEntropy loss;

    // Wahrscheinlichkeiten knapp bei 0/1 (soll nicht explodieren)
    auto p = from2D({
        {1.0 - 1e-12, 1e-12, 0.0},
        {1e-12, 1.0 - 1e-12, 0.0}
    });
    std::vector<int> y = {0, 1};

    double L = loss.forward(p, y);
    // Erwartung: kleiner, aber >0; insbesondere endlich
    assert(std::isfinite(L) && L >= 0.0);
    std::cout << "Clipping near 0/1 ✔\n";
}

void test_error_cases() {
    LossCategoricalCrossEntropy loss;

    // Shape-Mismatch (rows)
    auto p = from2D({{0.7,0.2,0.1}});
    std::vector<int> y_bad = {0, 2};
    expect_throw([&](){ (void)loss.forward(p, y_bad); }, "rows mismatch not detected");

    // Shape-Mismatch (one-hot)
    auto p2 = from2D({
        {0.7,0.2,0.1},
        {0.1,0.5,0.4}
    });
    auto y_onehot_bad = from2D({
        {1.0,0.0,0.0}
        // zweite Zeile fehlt
    });
    expect_throw([&](){ (void)loss.forward(p2, y_onehot_bad); }, "one-hot rows mismatch not detected");

    // Ungültiger Label-Index (nur wenn du das prüfst)
    // Erwartung: invalid_argument
    auto p3 = from2D({{0.7,0.2,0.1}});
    std::vector<int> y_oob = {5};
    // Wenn dieser Test bei dir (noch) nicht wirft, füge die Indexprüfung in deiner forward(labels) hinzu.
    // expect_throw([&](){ (void)loss.forward(p3, y_oob); }, "label index out of range not detected");

    std::cout << "Error cases ✔\n";
}

int main() {
    test_single_sample_values();
    test_two_sample_batch_label_and_onehot();
    test_clipping_edges();
    test_error_cases();

    std::cout << "All CCE forward checks passed ✅\n";
    return 0;
}
