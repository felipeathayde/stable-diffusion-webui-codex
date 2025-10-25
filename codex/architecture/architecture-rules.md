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
- Só engines/ e runtime/ através do façade `apps.server.backend` quando for público
- Internos podem importar de runtime submódulos diretamente quando necessário, sem criar ciclos

Assets
- Diffusers: organizar em subpastas padrão (tokenizer/, text_encoder[/_2]/, transformer|unet/, vae/, scheduler/); model_index.json presente
- WAN: High.gguf, Low.gguf, tokenizer/ (tokenizer.json, tokenizer_config.json, spiece.model), text_encoder/ (config.json + pesos), VAE (dir ou arquivo único)

Checks automáticos
- Não utilizamos validadores automáticos. Siga estritamente estas regras e os checklists nos docs.

Codex (substitui “Forge” no backend)
- Nomenclatura: use `CodexDiffusionEngine`/`CodexObjects` nos engines; evite introduzir novos campos `forge_*`. Aliases existem apenas para leitura de código legado durante a transição.
- APIs nativas Codex:
  - `apps.server.backend.codex.initialization.initialize_codex()` — bootstrap do ambiente (no‑op em setups mínimos).
  - `apps.server.backend.codex.main.{modules_change, checkpoint_change, refresh_model_loading_parameters}` — gestão de módulos/modelos via opts nativos.
  - `apps.server.backend.codex.options` — leitura de opções (`codex_*`), com fallback interno para nomes antigos apenas para leitura.
- Interação com A1111/legacy: se (e somente se) precisar tocar `modules.*`, importe via façade `apps.server.backend.modules.*`. Não use `modules.*` diretamente no backend ativo.
- Proibição: `modules_forge.*` não deve aparecer em código novo. Se houver necessidade funcional, exponha a capacidade via `apps.server.backend.codex.*` e implemente por trás conforme o plano de migração.
