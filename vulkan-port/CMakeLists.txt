cmake_minimum_required(VERSION 3.19)
project(lumina_q1_vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)

if(NOT Vulkan_GLSLC_EXECUTABLE)
  message(FATAL_ERROR "glslc not found - set VULKAN_SDK / CMAKE_PREFIX_PATH")
endif()

set(SHADERS q1_gemv_float q1_gemv_int8 q1_gemv_rp_float q1_gemv_rp_int8 q1_gemv_sk_float q1_gemv_sk_int8
            cm_bench_int8 cm_bench_f16 q1_gemm_int8_cm q1_gemv_rp_i8v q1_gemv_i8v_orig)
set(SPV_FILES "")
foreach(s ${SHADERS})
  set(src  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${s}.comp)
  set(spv  ${CMAKE_CURRENT_BINARY_DIR}/${s}.comp.spv)
  add_custom_command(
    OUTPUT  ${spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${src} -o ${spv} --target-env=vulkan1.1
    DEPENDS ${src}
    COMMENT "glslc ${s}.comp")
  list(APPEND SPV_FILES ${spv})
endforeach()

# Tunable strip-GEMV variants: "<name>|<define>|<define>..." ( '|' because CMake lists use ';' ).
set(TUNE_VARIANTS
  "tune_ns8|NSTRIP=8"
  "tune_ns16|NSTRIP=16"
  "tune_ns8v|NSTRIP=8|AVVEC=1"
  "tune_ns8f|NSTRIP=8|FAST=1"
  "tune_ns8vf|NSTRIP=8|AVVEC=1|FAST=1"
  "tune_ns16vf|NSTRIP=16|AVVEC=1|FAST=1"
  "tune_ns16vf_r2|NSTRIP=16|AVVEC=1|FAST=1|RPL=2"
  "tune_ns8vf_r2|NSTRIP=8|AVVEC=1|FAST=1|RPL=2"
  "tune_ns8vf_r4|NSTRIP=8|AVVEC=1|FAST=1|RPL=4"
  "tune_ns4vf_r2|NSTRIP=4|AVVEC=1|FAST=1|RPL=2"
  "tune_bw|NSTRIP=8|MODE=1"
  "tune_bw2|NSTRIP=8|MODE=2")
foreach(v ${TUNE_VARIANTS})
  string(REPLACE "|" ";" fields "${v}")
  list(GET fields 0 vname)
  list(REMOVE_AT fields 0)
  set(defs "")
  foreach(d ${fields})
    list(APPEND defs -D${d})
  endforeach()
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q1_gemv_tune.comp)
  set(spv ${CMAKE_CURRENT_BINARY_DIR}/${vname}.comp.spv)
  add_custom_command(
    OUTPUT  ${spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${src} -o ${spv} --target-env=vulkan1.1 ${defs}
    DEPENDS ${src}
    COMMENT "glslc ${vname}")
  list(APPEND SPV_FILES ${spv})
endforeach()

add_custom_target(lumina_q1_shaders ALL DEPENDS ${SPV_FILES})

add_executable(lumina_q1 src/main.cpp)
target_link_libraries(lumina_q1 PRIVATE Vulkan::Vulkan)
add_dependencies(lumina_q1 lumina_q1_shaders)

# put the compiled .spv next to the executable so the default run works
add_custom_command(TARGET lumina_q1 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_FILES}
          $<TARGET_FILE_DIR:lumina_q1>)
