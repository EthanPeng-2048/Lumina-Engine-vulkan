#pragma once

#include <cstdio>

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);

// Lumina-Engine project / developer links
const char * lumina_project_github(void);
const char * lumina_project_donate(void);
const char * lumina_developer_bilibili(void);

// Print the Lumina-Engine build identity and project links
void llama_print_lumina_banner(FILE * = stderr);

void llama_print_build_info(const char *, FILE * = stderr);
