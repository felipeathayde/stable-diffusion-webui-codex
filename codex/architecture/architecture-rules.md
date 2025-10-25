Architecture Rules — Do/Don’t
Date: 2025-10-25

Do
- Engines per tarefa: apps/server/backend/engines/diffusion/{txt2img,img2img,txt2vid,img2vid}.py
- Registrar engines em engines/registration.py (sem condicionais profundas por engine)
- Preferir PyTorch SDPA para atenção; sem kernels custom se o PyTorch já oferece
- WAN 2.2: usar runtime/nn/wan22.py; stages High/Low via GGUF + TE/Tok + VAE; erros explícitos
- UI: única fonte de blocos em apps/interface/blocks.json (+ overrides em blocks.d)

Don’t
- Não importar de legacy/ ou DEPRECATED/ do backend ativo
- Não usar `wan_gguf*` (qualquer pacote/módulo com esse nome)
- Não usar o namespace antigo `backend.*`; usar `apps.server.backend.*`
- Não criar engines/video/wan/*
- Não importar `modules_forge.*` no backend ativo (ver seção Codex abaixo)

Estrita/Políticas
- CODEX_STRICT_EXIT=1 (default): qualquer exceção sem tratamento encerra o processo após dump (logs/exceptions-YYYYmmdd-<pid>.log)
- Sem guardas automáticos de política: a disciplina é documental e de revisão. Violações (ex.: imports proibidos) devem ser rejeitadas em revisão de código.

Imports/Aliases
- Não use `modules.*` nem qualquer façade para ele dentro de `apps/server/backend/**`.
- Engines e runtime importam apenas código nativo sob `apps/server/backend/**` (evitar ciclos).

Assets and Registries
- Diffusers: organize in standard subfolders (tokenizer/, text_encoder[/_2]/, transformer|unet/, vae/, scheduler/) with model_index.json present.
- WAN: High.gguf, Low.gguf, tokenizer/ (tokenizer.json, tokenizer_config.json, spiece.model), text_encoder/ (config.json + weights), and a VAE.
- Do not rely on legacy discovery. The backend must expose native registries for VAEs, text encoders, and adapters. Discovery sources are explicit: local models/ and vendored HuggingFace trees under apps/server/backend/huggingface/.
- When an asset is missing, raise a clear error; never “guess” silently.

Checks automáticos
- Não utilizamos validadores automáticos. Siga estritamente estas regras e os checklists nos docs.

Codex (substitui “Forge” no backend)
- Nomenclatura: use `CodexDiffusionEngine`/`CodexObjects` nos engines; evite introduzir novos campos `forge_*`. Aliases existem apenas para leitura de código legado durante a transição.
- APIs nativas Codex:
  - `apps.server.backend.codex.initialization.initialize_codex()` — bootstrap do ambiente (no‑op em setups mínimos).
  - `apps.server.backend.codex.main.{modules_change, checkpoint_change, refresh_model_loading_parameters}` — gestão de módulos/modelos via opts nativos.
  - `apps.server.backend.codex.options` — leitura de opções (`codex_*`), com fallback interno para nomes antigos apenas para leitura.
- Interação com A1111/legacy: proibida no backend ativo. Se alguma capacidade ainda faltar, implemente nativamente sob `apps/server/backend/**` e exponha via `apps.server.backend.codex.*`.
- Proibição: `modules_forge.*` não deve aparecer em código novo. Se houver necessidade funcional, exponha a capacidade via `apps.server.backend.codex.*` e implemente por trás conforme o plano de migração.
