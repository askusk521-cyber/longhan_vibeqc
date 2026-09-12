#include "scf/cuda/arena.hpp"

#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Own checked resource calculations in an ordinary C++ translation unit; no device launch or
 * numerical kernel is compiled here. */
template <typename T>
bool append_array(std::size_t count, std::size_t& cursor, std::size_t& offset) {
  const std::size_t remainder = cursor % alignof(T);
  if (remainder != 0 && !checked_add(cursor, alignof(T) - remainder, cursor)) return false;
  offset = cursor;
  std::size_t bytes = 0;
  return checked_multiply(count, sizeof(T), bytes) && checked_add(cursor, bytes, cursor);
}

bool make_layout(std::size_t batch_size, std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                 std::size_t shell_count, std::size_t shell_pair_count,
                 std::size_t shell_pair_block_count, std::size_t bounded_generated_task_capacity,
                 std::size_t shell_pair_primitive_count, std::size_t psss_resident_task_count,
                 std::size_t psss_resident_ket_pair_count, std::size_t shell_quartet_tile_count,
                 std::size_t fp32_shell_quartet_tile_count,
                 std::size_t generated_shell_task_capacity,
                 std::size_t ppps_resident_ket_task_capacity,
                 std::size_t generic_order5_tile_capacity, std::size_t primitives,
                 std::size_t diis_history, std::size_t eigensolver_profile_capacity,
                 std::size_t spin_count, bool persistent_eri, bool transformed_direct,
                 bool shell_class_profiling, bool inactive_eigensolver_profiling,
                 bool bounded_fock_class_timing, bool bounded_direct_streaming,
                 bool mixed_precision_fock, ArenaLayout& layout) {
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrices = 0;
  std::size_t spin_matrices = 0;
  std::size_t eris = 0;
  std::size_t aos = 0;
  std::size_t direct_aos = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrices = 0;
  std::size_t direct_spin_matrices = 0;
  std::size_t transform_elements = 0;
  std::size_t transform_temporaries = 0;
  std::size_t ppps_signature_elements = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  if (!checked_multiply(nbf, nbf, matrix_size) ||
      !checked_multiply(matrix_size, matrix_size, eri_size) ||
      !checked_multiply(batch_size, matrix_size, matrices) ||
      !checked_multiply(matrices, spin_count, spin_matrices) ||
      !checked_multiply(batch_size, nbf, aos) ||
      !checked_multiply(batch_size, direct_nbf, direct_aos) ||
      !checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !checked_multiply(batch_size, direct_matrix_size, direct_matrices) ||
      !checked_multiply(direct_matrices, spin_count, direct_spin_matrices) ||
      !checked_multiply(aos, direct_nbf, transform_elements) ||
      !checked_multiply(transform_elements, spin_count, transform_temporaries) ||
      !checked_multiply(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
                        kPppsSignatureBucketCount, ppps_signature_elements) ||
      !checked_add(nbf, 1, nbf_plus_one) || !checked_multiply(nbf, nbf_plus_one, pair_product))
    return false;
  const std::size_t pair_count = pair_product / 2;
  if (persistent_eri && !checked_multiply(batch_size, eri_size, eris)) {
    return false;
  }
  std::size_t history_matrices = 0;
  std::size_t diis_dimension = 0;
  std::size_t diis_linear_elements = 0;
  if (!checked_multiply(spin_matrices, diis_history, history_matrices) ||
      !checked_add(diis_history, 1, diis_dimension) ||
      !checked_multiply(diis_dimension, diis_dimension, diis_linear_elements) ||
      !checked_multiply(diis_linear_elements, batch_size, diis_linear_elements))
    return false;
  std::size_t cursor = 0;
  ArenaLayout made{};
  if (!append_array<std::int64_t>(batch_size + 1, cursor, made.atom_offsets) ||
      !append_array<std::int32_t>(atoms, cursor, made.atom_systems) ||
      !append_array<std::int32_t>(atoms, cursor, made.atomic_numbers) ||
      !append_array<double>(atoms * 3, cursor, made.positions) ||
      !append_array<std::int64_t>(batch_size + 1, cursor, made.system_shell_offsets) ||
      !append_array<std::int32_t>(shell_count, cursor, made.shell_atoms) ||
      !append_array<std::uint8_t>(shell_count, cursor, made.shell_angular) ||
      !append_array<std::int64_t>(shell_count + 1, cursor, made.shell_ao_offsets) ||
      !append_array<std::int64_t>(shell_count + 1, cursor, made.shell_direct_ao_offsets) ||
      !append_array<std::int64_t>(shell_count + 1, cursor, made.shell_primitive_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor, made.system_shell_pair_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor, made.system_shell_quartet_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor, made.system_shell_pair_block_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor,
                                  made.system_shell_pair_block_quartet_offsets) ||
      !append_array<std::int32_t>(shell_pair_count, cursor, made.shell_pair_systems) ||
      !append_array<std::int32_t>(shell_pair_count, cursor, made.shell_pair_first) ||
      !append_array<std::int32_t>(shell_pair_count, cursor, made.shell_pair_second) ||
      !append_array<std::int64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : shell_pair_count + 1,
          cursor, made.shell_pair_primitive_offsets) ||
      !append_array<PrimitivePairData>(shell_quartet_tile_count == 0 && !bounded_direct_streaming
                                           ? 0
                                           : shell_pair_primitive_count,
                                       cursor, made.shell_primitive_pairs) ||
      !append_array<PsssResidentTask>(psss_resident_task_count, cursor, made.psss_resident_tasks) ||
      !append_array<std::uint32_t>(psss_resident_ket_pair_count, cursor,
                                   made.psss_resident_ket_pairs) ||
      !append_array<std::int32_t>(aos, cursor, made.ao_shells) ||
      !append_array<std::uint8_t>(aos, cursor, made.ao_term_counts) ||
      !append_array<std::uint8_t>(aos * kMaximumAoExpansionTerms * 3, cursor,
                                  made.ao_term_angular) ||
      !append_array<double>(aos * kMaximumAoExpansionTerms, cursor, made.ao_term_coefficients) ||
      !append_array<std::int32_t>(direct_aos, cursor, made.direct_ao_shells) ||
      !append_array<std::uint8_t>(direct_aos * 3, cursor, made.direct_ao_angular) ||
      !append_array<double>(direct_aos, cursor, made.direct_ao_coefficients) ||
      !append_array<double>(transformed_direct ? transform_elements : 0, cursor,
                            made.ao_to_direct_transform) ||
      !append_array<double>(primitives, cursor, made.primitive_exponents) ||
      !append_array<double>(primitives, cursor, made.primitive_coefficients) ||
      !append_array<std::int32_t>(batch_size * spin_count, cursor, made.occupied) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.warm_mask) ||
      !append_array<std::uint32_t>(mixed_precision_fock ? batch_size : 0, cursor,
                                   made.mixed_item_census) ||
      !append_array<double>(spin_matrices, cursor, made.warm_density) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.warm_invalid) ||
      !append_array<double>(matrices, cursor, made.overlap) ||
      !append_array<double>(matrices, cursor, made.hcore) ||
      !append_array<double>(eris, cursor, made.eri) ||
      !append_array<double>(persistent_eri ? 0 : (transformed_direct ? direct_matrices : matrices),
                            cursor, made.schwarz_bounds) ||
      !append_array<double>(transformed_direct ? direct_spin_matrices : 0, cursor,
                            made.direct_density) ||
      !append_array<double>(transformed_direct ? direct_spin_matrices : 0, cursor,
                            made.direct_fock) ||
      !append_array<double>(transformed_direct ? transform_temporaries : 0, cursor,
                            made.direct_transform_temporary) ||
      !append_array<double>(persistent_eri ? 0 : shell_pair_count, cursor,
                            made.shell_pair_bounds) ||
      !append_array<ShellPairDensityBounds>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : shell_pair_count, cursor,
          made.shell_pair_density_bounds) ||
      !append_array<std::uint32_t>(bounded_direct_streaming ? shell_pair_count : 0, cursor,
                                   made.bounded_direct_shell_pair_order) ||
      !append_array<std::uint32_t>(bounded_direct_streaming ? shell_pair_count : 0, cursor,
                                   made.bounded_stream_shell_pair_order) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectShellPairClassCount * (batch_size + 1) : 0,
          cursor, made.bounded_stream_pair_class_offsets) ||
      !append_array<GeneratedShellPairStream>(bounded_direct_streaming ? 1 : 0, cursor,
                                              made.bounded_stream_topology) ||
      !append_array<double>(bounded_direct_streaming ? shell_pair_block_count : 0, cursor,
                            made.bounded_direct_shell_pair_block_bounds) ||
      !append_array<double>(bounded_direct_streaming ? batch_size : 0, cursor,
                            made.bounded_direct_system_density_bounds) ||
      !append_array<double>(
          bounded_direct_streaming ? batch_size * detail::kDirectShellPairClassCount : 0, cursor,
          made.bounded_direct_system_pair_density_bounds) ||
      !append_array<GeneratedShellTask>(
          bounded_direct_streaming ? bounded_generated_task_capacity : 0, cursor,
          made.bounded_direct_generated_tasks) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_direct_generated_task_counts) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount + 1 : 0, cursor,
          made.bounded_direct_generated_task_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount + 1 : 0, cursor,
          made.bounded_direct_generated_retry_task_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_direct_generated_task_heads) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_direct_generated_overflow) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_direct_generated_retry_mask) ||
      !append_array<std::uint32_t>(bounded_direct_streaming ? 1 : 0, cursor,
                                   made.bounded_direct_generated_retry_any) ||
      !append_array<std::uint32_t>(bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
                                   cursor, made.bounded_force_signature_counts) ||
      !append_array<std::uint32_t>(bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
                                   cursor, made.bounded_force_signature_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureScanBlockCount : 0, cursor,
          made.bounded_force_signature_block_offsets) ||
      !append_array<std::uint64_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_fock_class_timer_starts) ||
      !append_array<std::uint64_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_fock_class_timer_elapsed) ||
      !append_array<std::uint32_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_fock_class_timer_launches) ||
      !append_array<unsigned long long>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_fock_fp64_work_counts) ||
      !append_array<unsigned long long>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.bounded_fock_fp32_work_counts) ||
      !append_array<std::uint32_t>(persistent_eri || bounded_direct_streaming
                                       ? 0
                                       : detail::kDirectQuartetAngularOrderCount + 1,
                                   cursor, made.active_shell_quartet_tile_offsets) ||
      !append_array<std::uint32_t>(
          persistent_eri || bounded_direct_streaming ? 0 : detail::kDirectQuartetAngularOrderCount,
          cursor, made.active_shell_quartet_tile_counts) ||
      !append_array<ActiveShellQuartetTile>(
          persistent_eri || bounded_direct_streaming ? 0 : shell_quartet_tile_count, cursor,
          made.active_shell_quartet_tiles) ||
      !append_array<std::uint32_t>(
          mixed_precision_fock ? detail::kDirectQuartetAngularOrderCount + 1 : 0, cursor,
          made.fp32_shell_quartet_tile_offsets) ||
      !append_array<std::uint32_t>(
          mixed_precision_fock ? detail::kDirectQuartetAngularOrderCount : 0, cursor,
          made.fp32_shell_quartet_tile_counts) ||
      !append_array<ActiveShellQuartetTile>(
          mixed_precision_fock ? fp32_shell_quartet_tile_count : 0, cursor,
          made.fp32_shell_quartet_tiles) ||
      !append_array<DeviceShellClassProfileEntry>(
          shell_class_profiling ? detail::kDirectQuartetShellClassCount : 0, cursor,
          made.shell_class_profile) ||
      !append_array<std::uint32_t>(
          shell_quartet_tile_count == 0 ? 0 : kPersistentFockAngularOrderCount, cursor,
          made.persistent_fock_task_heads) ||
      !append_array<std::uint32_t>(mixed_precision_fock ? kPersistentFockAngularOrderCount : 0,
                                   cursor, made.fp32_persistent_fock_task_heads) ||
      !append_array<std::uint32_t>(
          shell_quartet_tile_count == 0 ? 0 : kPersistentForceAngularOrderCount, cursor,
          made.persistent_force_task_heads) ||
      !append_array<GeneratedShellTask>(generated_shell_task_capacity, cursor,
                                        made.generated_shell_tasks) ||
      !append_array<std::uint8_t>(generated_shell_task_capacity == 0 ? 0 : shell_quartet_tile_count,
                                  cursor, made.generated_shell_classes) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount + 1,
          cursor, made.generated_shell_task_offsets) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount, cursor,
          made.generated_shell_task_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount, cursor,
          made.generated_shell_task_write_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount, cursor,
          made.generated_shell_task_heads) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : kLowOrderSignatureElementCount, cursor,
          made.generated_low_order_signature_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : kLowOrderSignatureElementCount, cursor,
          made.generated_low_order_signature_offsets) ||
      !append_array<GeneratedPppsResidentTask>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count, cursor,
          made.generated_ppps_resident_tasks) ||
      !append_array<std::uint32_t>(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
                                   cursor, made.generated_ppps_resident_bra_counts) ||
      !append_array<std::uint32_t>(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count + 1,
                                   cursor, made.generated_ppps_resident_bra_offsets) ||
      !append_array<std::uint32_t>(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
                                   cursor, made.generated_ppps_resident_bra_write_counts) ||
      !append_array<std::uint32_t>(ppps_signature_elements, cursor,
                                   made.generated_ppps_resident_signature_counts) ||
      !append_array<std::uint32_t>(ppps_signature_elements, cursor,
                                   made.generated_ppps_resident_signature_offsets) ||
      !append_array<std::uint32_t>(shell_class_profiling ? ppps_resident_ket_task_capacity : 0,
                                   cursor, made.generated_ppps_resident_signatures) ||
      !append_array<std::uint64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : 1, cursor,
          made.generated_fock_shell_class_mask) ||
      !append_array<std::uint64_t>(mixed_precision_fock ? 1 : 0, cursor,
                                   made.generated_mixed_fock_shell_class_mask) ||
      !append_array<ActiveShellQuartetTile>(generic_order5_tile_capacity, cursor,
                                            made.generic_order5_tiles) ||
      !append_array<std::uint32_t>(generic_order5_tile_capacity == 0 ? 0 : 1, cursor,
                                   made.generic_order5_tile_count) ||
      !append_array<std::int32_t>(pair_count, cursor, made.ao_pair_first) ||
      !append_array<std::int32_t>(pair_count, cursor, made.ao_pair_second) ||
      !append_array<double>(batch_size, cursor, made.nuclear_repulsion) ||
      !append_array<double>(matrices, cursor, made.orthogonalizer) ||
      !append_array<double>(spin_matrices, cursor, made.temporary) ||
      !append_array<double>(spin_matrices, cursor, made.eigensystem) ||
      !append_array<double>(spin_matrices, cursor, made.coefficients) ||
      !append_array<double>(batch_size * spin_count * nbf, cursor, made.eigenvalues) ||
      !append_array<double>(spin_matrices, cursor, made.density) ||
      !append_array<double>(spin_matrices, cursor, made.next_density) ||
      !append_array<double>(spin_matrices, cursor, made.fock) ||
      !append_array<double>(spin_matrices, cursor, made.residual) ||
      !append_array<double>(spin_matrices, cursor, made.weighted_density) ||
      !append_array<double>(spin_count == 2 ? matrices : 0, cursor, made.total_density) ||
      !append_array<double>(spin_count == 2 ? matrices : 0, cursor, made.total_weighted_density) ||
      !append_array<double>(history_matrices, cursor, made.fock_history) ||
      !append_array<double>(history_matrices, cursor, made.residual_history) ||
      !append_array<double>(diis_linear_elements, cursor, made.diis_linear_system) ||
      !append_array<double>(batch_size * diis_dimension, cursor, made.diis_coefficients) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.diis_count) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.diis_head) ||
      !append_array<double>(batch_size, cursor, made.energy) ||
      !append_array<double>(batch_size, cursor, made.previous_energy) ||
      !append_array<double>(batch_size, cursor, made.energy_change) ||
      !append_array<double>(batch_size, cursor, made.density_rms) ||
      !append_array<double>(atoms * 3, cursor, made.forces) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.active) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.converged) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.failed) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.final_fock_reuse_mask) ||
      !append_array<std::uint32_t>(1, cursor, made.final_fock_rebuild_count) ||
      !append_array<std::uint8_t>(batch_size * spin_count, cursor, made.spin_active) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.iterations) ||
      !append_array<int>(batch_size * spin_count, cursor, made.solver_info) ||
      !append_array<std::uint32_t>(inactive_eigensolver_profiling ? 1 : 0, cursor,
                                   made.inactive_eigensolver_profile_count) ||
      !append_array<DeviceInactiveEigensolverProfileEntry>(
          inactive_eigensolver_profiling ? eigensolver_profile_capacity : 0, cursor,
          made.inactive_eigensolver_profile) ||
      !append_array<std::uint64_t>(bounded_direct_streaming ? 1 : 0, cursor,
                                   made.bounded_direct_cursor))
    return false;
  made.bytes = cursor;
  layout = made;
  return true;
}

}  // namespace vibeqc::scf::cuda_execution
