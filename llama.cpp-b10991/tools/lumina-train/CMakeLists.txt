# Lumina-Engine QLoRA fine-tuning tool
#
# Trains low-rank adapters on a frozen (possibly quantized) GGUF base model and
# writes a standard LoRA adapter GGUF. See lumina-train.cpp for details.

set(TARGET llama-train)

add_executable(${TARGET} lumina-train.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
