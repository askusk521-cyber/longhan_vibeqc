#include "scf/solver/diis.hpp"

#include <utility>

#include "runtime/resource_usage.hpp"

namespace vibeqc::scf::solver {
using reference::dot;
using reference::index;
using reference::solve_linear;
Diis::Diis(std::size_t capacity) : capacity_(capacity) {}

/** Actual retained numerical capacity; no temporary extrapolation work. */
std::size_t Diis::numeric_capacity() const noexcept {
  std::size_t bytes = 0;
  for (const auto& value : focks_)
    bytes = runtime::add_capacity(bytes, runtime::vector_bytes(value));
  for (const auto& value : residuals_)
    bytes = runtime::add_capacity(bytes, runtime::vector_bytes(value));
  return bytes;
}

void Diis::clear() {
  focks_.clear();
  residuals_.clear();
}

Matrix Diis::update(const Matrix& fock, const Matrix& residual) {
  if (capacity_ < 2) return fock;
  focks_.push_back(fock);
  residuals_.push_back(residual);
  if (focks_.size() > capacity_) {
    focks_.erase(focks_.begin());
    residuals_.erase(residuals_.begin());
  }
  if (focks_.size() < 2) return fock;

  const std::size_t m = focks_.size();
  const std::size_t dim = m + 1;
  Matrix b(dim * dim, 0.0);
  std::vector<double> rhs(dim, 0.0);
  rhs[m] = -1.0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      b[index(i, j, dim)] = dot(residuals_[i], residuals_[j]);
    }
    b[index(i, m, dim)] = -1.0;
    b[index(m, i, dim)] = -1.0;
  }
  std::vector<double> coefficients;
  if (!solve_linear(std::move(b), std::move(rhs), coefficients, dim)) return fock;

  Matrix extrapolated(fock.size(), 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t element = 0; element < fock.size(); ++element) {
      extrapolated[element] += coefficients[i] * focks_[i][element];
    }
  }
  return extrapolated;
}

}  // namespace vibeqc::scf::solver
