#ifndef VIBEQC_SCF_SOLVER_DIIS_HPP
#define VIBEQC_SCF_SOLVER_DIIS_HPP
#include "scf/reference/linalg.hpp"

namespace vibeqc::scf::solver {
using reference::Matrix;
/** Bounded conventional DIIS history for one trajectory, shared by both spins.
 * Singular augmented solves return the unextrapolated Fock. A proposal/reset
 * clears trajectory history; history never belongs to another batch item.
 */
class Diis {
 public:
  explicit Diis(std::size_t capacity);
  /** Retained numerical capacity, excluding temporary extrapolation storage. */
  std::size_t numeric_capacity() const noexcept;
  void clear();
  Matrix update(const Matrix& fock, const Matrix& residual);

 private:
  std::size_t capacity_;
  std::vector<Matrix> focks_;
  std::vector<Matrix> residuals_;
};
}  // namespace vibeqc::scf::solver
#endif
