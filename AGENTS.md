Listen.

This repo intentionally keeps `AGENTS.md` repo-specific. Global, repo-agnostic guidance lives in your Codex CLI `developer_instructions` (configured in your local Codex config, e.g. `~/.codex/config.toml`).

If something here conflicts with higher-priority system/developer instructions, follow the higher-priority instructions. Nested `AGENTS.md` files override this one within their directory subtree.

---

### ACT I – GIT (repo-specific additions)

Large artifacts, outputs, caches, and heavy model directories stay untracked per `gitignore.md`.
You obey the ignore and attributes policy in `gitignore.md`.

---

### ACT II – ARCHITECTURE, MODELS, PYTHON (repo-specific additions)

* The default core for attention is PyTorch SDPA (but we also support xFormers and SAGE).
* You read the archived upstream snapshots with a cold eye.
* Archived upstream snapshots are museums. Read only. If you need a behavior baseline, start with Hugging Face Diffusers — then come back to the museum if you still have to.

You do not keep or copy archived upstream snapshot code to `apps`.
You redesign in Codex style:
* Dataclasses and enums.
* Small modules with clear seams.
* Explicit errors.
* Readable names.

* Do not add Python shebangs.
* Do not create silent fallbacks: fail fast with clear errors.

Imports outside `/apps` are banned.
Only `apps.*` lives in active code.

If a feature has not been ported, you raise:
```python
NotImplementedError("<feature> not yet ported")
```

You follow the order for any implementation:
1. Understand the goal.
2. Search the equivalents in archived upstream snapshots and inspect.
3. Draft a native plan without copying code.
4. Then and only then write.

Model loading is a minefield you cross with a map.
You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`.

You prefer SafeTensors.
You call `torch.load(..., weights_only=True, mmap=True)` when it applies.

You treat GGUF the right way: bake or dequantize once before sampling, not every time like a fool.

If your script needs access to `~/work/stable-diffusion-webui-codex`, you set the path correctly:

```bash
PYTHONPATH=$HOME/work/stable-diffusion-webui-codex
```

---

### ACT III – FRONTEND, LAYOUT, AND CSS (repo-specific additions)

If you want to change anything in `apps/interface/src/styles`, you read that folder's `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.

---

### ACT IV – WORKFLOW, CHECKLISTS, AND DOCS (repo-specific additions)

* Treat `.sangoi/` as the primary source of truth:
  * `planning/` for plans, `task-logs/` for execution logs, `handoffs/` for handoffs, `.tools/` for internal tooling.
  * If a user asks for a handoff, consult `.sangoi/handoffs/HANDOFF_GUIDE.md`.
* Engine/runtime changes must follow a strict checklist (memory, conditioning, parity):
  * Smart offload: if you introduce a new transformer core, ensure it participates in the smart-offload policy (avoid OOM during assembly).
  * External text encoders / VAE: wrap with patchers and load/unload via the memory manager; no silent fallbacks.
  * Parity (“golesma”): validate sigma schedule, timestep semantics, sign conventions, stream/token order (RoPE), norm order, and integration dtype.
  * When a new parity root cause is found, record it in `.sangoi/task-logs/**` or `.sangoi/handoffs/**`.
* When you add/rename/restructure a directory (or change its responsibilities), update that directory’s `AGENTS.md` and keep `.sangoi/index/AGENTS-INDEX.md` in sync.
