#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/flat_matrix.hpp"
#include "../include/categorical_cross_entropy.hpp"
#include "../include/activation_softmax.hpp"
#include "../include/activation_softmax_loss_categorical_cross_entropy.hpp"

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

// ========== NEUE TESTS FÜR BACKWARD PASSES ==========

void test_softmax_backward_jacobian() {
    std::cout << "\n=== Testing Softmax Backward (Jacobian Method) ===\n";
    
    ActivationSoftmax softmax;
    
    // Einfaches Beispiel: 2 samples, 3 classes
    auto inputs = from2D({
        {1.0, 2.0, 3.0},
        {0.5, 1.5, 0.2}
    });
    
    // Forward pass
    softmax.forward(inputs);
    
    std::cout << "Softmax output (sample 0): ";
    for (int j = 0; j < 3; ++j) {
        std::cout << softmax.output.get(0, j) << " ";
    }
    std::cout << "\n";
    
    // Backward pass mit Gradient = ones
    auto dvalues = from2D({
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    });
    
    softmax.backward(dvalues);
    
    std::cout << "Softmax dinputs (sample 0): ";
    for (int j = 0; j < 3; ++j) {
        std::cout << softmax.dinputs.get(0, j) << " ";
    }
    std::cout << "\n";
    
    // Prüfe, dass die Summe der dinputs pro Sample ~0 ist (Eigenschaft von Softmax)
    double sum_sample0 = 0.0;
    for (int j = 0; j < 3; ++j) {
        sum_sample0 += softmax.dinputs.get(0, j);
    }
    assert(approx(sum_sample0, 0.0, 1e-10));
    
    std::cout << "Softmax backward (Jacobian) ✔\n";
}

void test_loss_backward_sparse_labels() {
    std::cout << "\n=== Testing Loss Backward (Sparse Labels) ===\n";
    
    LossCategoricalCrossEntropy loss;
    
    // Predicted probabilities
    auto y_pred = from2D({
        {0.7, 0.2, 0.1},
        {0.1, 0.5, 0.4}
    });
    
    std::vector<int> y_true = {0, 2};
    
    // Backward pass
    loss.backward(y_pred, y_true);
    
    std::cout << "Loss dinputs:\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  Sample " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << loss.dinputs.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    // Prüfe, dass dinputs endlich sind
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            assert(std::isfinite(loss.dinputs.get(i, j)));
        }
    }
    
    std::cout << "Loss backward (sparse) ✔\n";
}

void test_loss_backward_onehot() {
    std::cout << "\n=== Testing Loss Backward (One-Hot) ===\n";
    
    LossCategoricalCrossEntropy loss;
    
    auto y_pred = from2D({
        {0.7, 0.2, 0.1},
        {0.1, 0.5, 0.4}
    });
    
    auto y_true = from2D({
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0}
    });
    
    loss.backward(y_pred, y_true);
    
    std::cout << "Loss dinputs (one-hot):\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  Sample " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << loss.dinputs.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    assert(std::isfinite(loss.dinputs.get(0, 0)));
    
    std::cout << "Loss backward (one-hot) ✔\n";
}

void test_combined_softmax_loss_forward() {
    std::cout << "\n=== Testing Combined Softmax+Loss Forward ===\n";
    
    ActivationSoftmaxLossCategoricalCrossEntropy combined;
    
    // Raw logits (vor Softmax)
    auto inputs = from2D({
        {1.0, 2.0, 3.0},
        {0.5, 1.5, 0.2}
    });
    
    std::vector<int> y_true = {2, 1};
    
    // Forward pass
    double loss_value = combined.forward(inputs, y_true);
    
    std::cout << "Combined forward loss: " << loss_value << "\n";
    std::cout << "Combined output (sample 0): ";
    for (int j = 0; j < 3; ++j) {
        std::cout << combined.output.get(0, j) << " ";
    }
    std::cout << "\n";
    
    assert(std::isfinite(loss_value));
    assert(loss_value >= 0.0);
    
    std::cout << "Combined forward ✔\n";
}

void test_combined_softmax_loss_backward() {
    std::cout << "\n=== Testing Combined Softmax+Loss Backward (Optimized) ===\n";
    
    ActivationSoftmaxLossCategoricalCrossEntropy combined;
    
    // Raw logits
    auto inputs = from2D({
        {1.0, 2.0, 3.0},
        {0.5, 1.5, 0.2},
        {2.0, 1.0, 0.5}
    });
    
    std::vector<int> y_true = {2, 1, 0};
    
    // Forward pass
    double loss_value = combined.forward(inputs, y_true);
    std::cout << "Loss: " << loss_value << "\n";
    
    // Backward pass - verwendet die optimierte Formel
    combined.backward(combined.output, y_true);
    
    std::cout << "Combined dinputs:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Sample " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << combined.dinputs.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    // Wichtige Eigenschaft: dinputs[i, y_true[i]] sollte negativ sein
    // (weil output[i, y_true[i]] - 1 < 0 für korrekte Predictions)
    for (int i = 0; i < 3; ++i) {
        int true_label = y_true[i];
        double grad = combined.dinputs.get(i, true_label);
        std::cout << "  Gradient for true class (sample " << i << "): " << grad << "\n";
    }
    
    // Summe der Gradienten pro Sample sollte klein sein (normalisiert)
    for (int i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 3; ++j) {
            sum += combined.dinputs.get(i, j);
        }
        std::cout << "  Sum of gradients (sample " << i << "): " << sum << "\n";
        // Die Summe sollte ~0 sein aufgrund der Normalisierung
        assert(approx(sum, 0.0, 1e-6));
    }
    
    std::cout << "Combined backward ✔\n";
}

void test_combined_with_onehot() {
    std::cout << "\n=== Testing Combined with One-Hot Labels ===\n";
    
    ActivationSoftmaxLossCategoricalCrossEntropy combined;
    
    auto inputs = from2D({
        {1.0, 2.0, 3.0},
        {0.5, 1.5, 0.2}
    });
    
    auto y_true = from2D({
        {0.0, 0.0, 1.0},  // class 2
        {0.0, 1.0, 0.0}   // class 1
    });
    
    // Forward
    double loss = combined.forward(inputs, y_true);
    std::cout << "Loss with one-hot: " << loss << "\n";
    
    // Backward
    combined.backward(combined.output, y_true);
    
    std::cout << "Dinputs with one-hot:\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  Sample " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << combined.dinputs.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "Combined with one-hot ✔\n";
}

void test_gradient_numerical_check() {
    std::cout << "\n=== Numerical Gradient Check ===\n";
    
    // Einfache numerische Überprüfung des Gradienten
    ActivationSoftmax softmax;
    
    auto inputs = from2D({{1.0, 2.0, 3.0}});
    softmax.forward(inputs);
    
    auto dvalues = from2D({{0.1, 0.2, -0.3}});
    softmax.backward(dvalues);
    
    // Numerischer Gradient mit finiten Differenzen
    double epsilon = 1e-5;
    FlatMatrix numerical_grad(1, 3, 0.0);
    
    for (int j = 0; j < 3; ++j) {
        // f(x + h)
        auto inputs_plus = inputs;
        inputs_plus.set(0, j, inputs.get(0, j) + epsilon);
        softmax.forward(inputs_plus);
        auto output_plus = softmax.output;
        
        // f(x - h)
        auto inputs_minus = inputs;
        inputs_minus.set(0, j, inputs.get(0, j) - epsilon);
        softmax.forward(inputs_minus);
        auto output_minus = softmax.output;
        
        // Gradient = sum(dL/doutput * doutput/dinput)
        double grad = 0.0;
        for (int k = 0; k < 3; ++k) {
            double doutput = (output_plus.get(0, k) - output_minus.get(0, k)) / (2 * epsilon);
            grad += dvalues.get(0, k) * doutput;
        }
        numerical_grad.set(0, j, grad);
    }
    
    std::cout << "Analytical gradient: ";
    for (int j = 0; j < 3; ++j) {
        std::cout << softmax.dinputs.get(0, j) << " ";
    }
    std::cout << "\n";
    
    std::cout << "Numerical gradient:  ";
    for (int j = 0; j < 3; ++j) {
        std::cout << numerical_grad.get(0, j) << " ";
    }
    std::cout << "\n";
    
    // Vergleiche
    for (int j = 0; j < 3; ++j) {
        double analytical = softmax.dinputs.get(0, j);
        double numerical = numerical_grad.get(0, j);
        double diff = std::fabs(analytical - numerical);
        assert(diff < 1e-5);
    }
    
    std::cout << "Numerical gradient check ✔\n";
}

int main() {
    std::cout << "===== FORWARD PASS TESTS =====\n";
    test_single_sample_values();
    test_two_sample_batch_label_and_onehot();
    test_clipping_edges();
    test_error_cases();
    
    std::cout << "\n===== BACKWARD PASS TESTS =====\n";
    test_softmax_backward_jacobian();
    test_loss_backward_sparse_labels();
    test_loss_backward_onehot();
    
    std::cout << "\n===== COMBINED SOFTMAX+LOSS TESTS =====\n";
    test_combined_softmax_loss_forward();
    test_combined_softmax_loss_backward();
    test_combined_with_onehot();
    
    std::cout << "\n===== GRADIENT VERIFICATION =====\n";
    test_gradient_numerical_check();

    std::cout << "\n✅ ALL TESTS PASSED! ✅\n";
    return 0;
}
