#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *current_query = rater.GetNextQuery();

    // Build K_stack (0..i) and V_stack (0..i) in HBM
    Matrix *k_stack = matrix_memory_allocator.Allocate("k_stack_base");
    gpu_sim.Copy(keys[0], k_stack, Position::kInGpuHbm);
    Matrix *v_stack = matrix_memory_allocator.Allocate("v_stack_base");
    gpu_sim.Copy(values[0], v_stack, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *k_next = matrix_memory_allocator.Allocate("k_stack_next");
      gpu_sim.Concat(k_stack, keys[j], k_next, /*axis=*/0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_stack);
      k_stack = k_next;
      Matrix *v_next = matrix_memory_allocator.Allocate("v_stack_next");
      gpu_sim.Concat(v_stack, values[j], v_next, /*axis=*/0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(v_stack);
      v_stack = v_next;
    }

    // Move operands for compute to SRAM and transpose K to K^T
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_stack);
    gpu_sim.MoveMatrixToSharedMem(v_stack);
    gpu_sim.Transpose(k_stack, Position::kInSharedMemory);

    // Compute row-wise to minimize SRAM peak
    Matrix *output = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *q_row = matrix_memory_allocator.Allocate("q_row");
      gpu_sim.GetRow(current_query, r, q_row, Position::kInSharedMemory);

      Matrix *row_logits = matrix_memory_allocator.Allocate("row_logits");
      gpu_sim.MatMul(q_row, k_stack, row_logits); // (1,i+1)

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_logits, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      Matrix *row_out = matrix_memory_allocator.Allocate("row_out");
      gpu_sim.MatMul(row_soft, v_stack, row_out); // (1,512)

      if (r == 0) {
        output = row_out;
      } else {
        Matrix *new_output = matrix_memory_allocator.Allocate("output_cat");
        gpu_sim.Concat(output, row_out, new_output, /*axis=*/0,
                       Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(output);
        gpu_sim.ReleaseMatrix(row_out);
        output = new_output;
      }
      gpu_sim.ReleaseMatrix(q_row);
      gpu_sim.ReleaseMatrix(row_logits);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
    }

    gpu_sim.ReleaseMatrix(k_stack);
    gpu_sim.ReleaseMatrix(v_stack);

    gpu_sim.MoveMatrixToGpuHbm(output);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*output);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
