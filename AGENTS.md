# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- **What this repo is**: a sequenced CUDA curriculum (`curriculum/`) plus runnable samples (`code/`) plus honestly-dated resource curation (`resources/`). See the root `README.md` for the pitch and `curriculum/README.md` for the stage table.
- **No GPU in this environment**: samples are compile-checked with Docker, not run. `docker run --rm -v "$PWD:/work" -w /work/code/0N-sample-name nvidia/cuda:12.6.2-devel-ubuntu22.04 make` is the standard check; CI (`.github/workflows/ci.yml`) does the same for every sample under `code/*/Makefile`. Never claim a sample was run or its performance numbers measured unless you actually ran it on real hardware — see `code/README.md`'s "GPU access" section for the exact wording this repo uses about that gap.
- **Adding a sample**: one `.cu` file + `Makefile` + `README.md` per directory under `code/`, following the existing numbered-stage naming (`0N-topic-name`). Reuse the `-gencode` arch list (sm_70/75/80/86/89) and `code/common/check.cuh`'s `CUDA_CHECK`/`CUDA_CHECK_LAST` macros already used by every sample — don't invent a new error-checking pattern.
- **Citation discipline**: every external link in `curriculum/*.md` and `resources/*.md` must be a URL someone actually opened and confirmed live — this was the explicit standard the repo was built to, and it's why `resources/curated-resources.md` states dates and "aged/current" verdicts per entry. Before adding or changing a citation, fetch the URL yourself; don't restate one from memory.
- **Link hygiene**: internal markdown links are checked with a small ad-hoc script during authoring (walk all `.md` files, resolve relative link targets against disk) rather than a checked-in tool — there's no `package.json`/lint step for this, so re-verify manually after renaming or moving a doc.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
