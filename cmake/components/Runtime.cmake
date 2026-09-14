include_guard(GLOBAL)

function(vibeqc_add_runtime_sources target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "vibeqc_add_runtime_sources target does not exist: ${target}")
  endif()

  target_sources("${target}" PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/context.cpp")

  if(VIBEQC_ENABLE_CUDA)
    target_sources("${target}" PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/cuda_runtime.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/cuda_component_trace.cpp")
  endif()
endfunction()
