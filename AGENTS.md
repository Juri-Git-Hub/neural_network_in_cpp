# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/`, one component per translation unit (e.g., `matrix.cpp`, `layer_dense.cpp`).
- Public interfaces stay in `include/`; keep headers lightweight and include only what you use.
- The `build/` directory stores generated artifacts, including the `neural_network_cpp` executable.
- Document new datasets or tooling folders in `README.md` for traceability.

## Build, Test, and Development Commands
- `cmake -S . -B build`: configure the project with CMake 3.16+ and C++17 defaults.
- `cmake --build build`: compile all targets; reruns only the files that changed.
- `./build/neural_network_cpp`: execute the sample run to inspect forward passes or training logs.
- `cmake --build build --target clean`: remove cached objects when switching toolchains or branches.

## Coding Style & Naming Conventions
- Follow the current two-space indentation and brace-on-same-line layout used across `src/`.
- Classes use PascalCase (`Matrix`, `LayerDense`); functions, variables, and free helpers use lowerCamelCase.
- Prefix member fields with `m_` and keep helper functions in anonymous namespaces when the linkage is local.
- Use `#pragma once` in headers and prefer quoted includes for project headers (`"matrix.hpp"`).

## Testing Guidelines
- No automated suite exists yet; add unit tests under a new `tests/` tree with GoogleTest or lightweight assertions.
- Register new tests through CMake (`add_executable` + `add_test`) so they run via `ctest`.
- Until tests land, exercise key paths by running `./build/neural_network_cpp` with representative sample data and log the results in PRs.

## Commit & Pull Request Guidelines
- Commit messages follow focused, present-tense summaries similar to `added activation softmax class`.
- Use topic branches (`feature/softmax-loss`, `fix/matrix-bounds`) and keep each commit buildable.
- Pull requests should explain the change set, list manual or automated test evidence, and link related issues or TODOs.
- Attach screenshots or console excerpts when behavior changes (e.g., accuracy metrics) to speed reviewer context.

## Agent Support Principles
- Prioritize learning: explain underlying math, C++ design choices, and trade-offs instead of dropping turnkey fixes.
- Coach through solutions by suggesting strategies, asking guiding questions, and pointing to relevant files or equations.
- Highlight best practices for scaling C++ projects (modularity, RAII, testing) and tie them back to neural network goals.
- Avoid providing prebuilt libraries or large pasted solutions; instead, break work into steps the maintainer can implement.
- Encourage reflection: summarize what changed, why it matters for understanding neural networks, and suggest next study topics.
