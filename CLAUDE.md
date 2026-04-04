# Claude Code Instructions

## Session Start
- Read `KNOWLEDGE_BASE.md` at the beginning of each session to understand the project structure, backends, benchmark modes, and design decisions.
- If the user references specific components (backends, benchmark modes, plotter, etc.), use the knowledge base as context before diving into code.

## Session End
- Before ending a session where code or architecture was changed, review `KNOWLEDGE_BASE.md` and update any sections that are now outdated (new files, changed interfaces, new backends, modified config, etc.).
- Keep the knowledge base concise and accurate. Don't let it drift from the actual codebase.

## Project Context
- This is a Linux-targeted I/O benchmarking project. The C++ backend and python_self_backend use POSIX-only APIs.
- The goal is to improve I/O throughput times through algorithm evolution and experimentation.
- Kubernetes deployment is optional; the benchmarks can run on any Linux machine with sufficient RAM.
- The C++ extension is built via `python setup.py build_ext --inplace`.

## Code Style
- Backends expose async functions: `{name}_read_blocks(block_size, buffer, block_indices, dest_files) -> float` and `{name}_write_blocks(...)` returning elapsed seconds.
- All writes use the atomic temp-file-then-rename pattern.
- Results are JSON files saved incrementally with checkpoint/resume support.

## Code Changes
- Do not modify code without the user's explicit approval. Always show proposed changes first and wait for confirmation before applying.

## Git Commits
- Do not add "Co-Authored-By" lines to commit messages.
