include_guard(GLOBAL)
include(CMakeParseArguments)

# Register a deterministic generated-source family without hiding its build graph.
#
# Required:
#   NAME       Custom target name for the generated family.
#   TARGET     Owning build target; receives an explicit dependency on NAME.
#   GENERATOR  Generator script invoked through Python3.
#   OUTPUTS    Complete list of declared outputs.
#
# Optional:
#   BYPRODUCTS       Additional declared byproducts.
#   DEPENDS          Generator/spec/input dependencies.
#   ARGS             Generator command-line arguments.
#   LANGUAGE         CMake LANGUAGE property for generated outputs.
#   COMPILE_OPTIONS  Per-source compile options for generated outputs.
#   ADD_TO_TARGET    Attach OUTPUTS to TARGET with target_sources().
function(vibeqc_register_generated_sources)
  set(options ADD_TO_TARGET)
  set(one_value_args NAME TARGET GENERATOR LANGUAGE)
  set(multi_value_args OUTPUTS BYPRODUCTS DEPENDS ARGS COMPILE_OPTIONS)
  cmake_parse_arguments(VGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  foreach(required IN ITEMS NAME TARGET GENERATOR)
    if(NOT VGS_${required})
      message(FATAL_ERROR "vibeqc_register_generated_sources requires ${required}")
    endif()
  endforeach()
  if(NOT VGS_OUTPUTS)
    message(FATAL_ERROR "vibeqc_register_generated_sources requires OUTPUTS")
  endif()
  if(NOT TARGET "${VGS_TARGET}")
    message(FATAL_ERROR
            "vibeqc_register_generated_sources target does not exist: ${VGS_TARGET}")
  endif()
  if(TARGET "${VGS_NAME}")
    message(FATAL_ERROR
            "vibeqc_register_generated_sources target already exists: ${VGS_NAME}")
  endif()

  add_custom_command(
    OUTPUT ${VGS_OUTPUTS}
    BYPRODUCTS ${VGS_BYPRODUCTS}
    COMMAND "${Python3_EXECUTABLE}" "${VGS_GENERATOR}" ${VGS_ARGS}
    DEPENDS "${VGS_GENERATOR}" ${VGS_DEPENDS}
    VERBATIM)
  add_custom_target("${VGS_NAME}" DEPENDS ${VGS_OUTPUTS})
  add_dependencies("${VGS_TARGET}" "${VGS_NAME}")

  set_source_files_properties(${VGS_OUTPUTS} ${VGS_BYPRODUCTS}
                              PROPERTIES GENERATED TRUE)
  if(VGS_LANGUAGE)
    set_source_files_properties(${VGS_OUTPUTS} PROPERTIES LANGUAGE "${VGS_LANGUAGE}")
  endif()
  if(VGS_COMPILE_OPTIONS)
    set_source_files_properties(${VGS_OUTPUTS}
                                PROPERTIES COMPILE_OPTIONS "${VGS_COMPILE_OPTIONS}")
  endif()
  if(VGS_ADD_TO_TARGET)
    target_sources("${VGS_TARGET}" PRIVATE ${VGS_OUTPUTS})
  endif()
endfunction()
