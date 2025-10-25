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

Estrita/Políticas
- CODEX_STRICT_EXIT=1 (default): qualquer exceção sem tratamento encerra o processo após dump (logs/exceptions-YYYYmmdd-<pid>.log)
- CODEX_POLICY_STRICT=1 (default): violação de política (imports proibidos) encerra no startup

Imports/Aliases
- Só engines/ e runtime/ através do façade `apps.server.backend` quando for público
- Internos podem importar de runtime submódulos diretamente quando necessário, sem criar ciclos

Assets
- Diffusers: organizar em subpastas padrão (tokenizer/, text_encoder[/_2]/, transformer|unet/, vae/, scheduler/); model_index.json presente
- WAN: High.gguf, Low.gguf, tokenizer/ (tokenizer.json, tokenizer_config.json, spiece.model), text_encoder/ (config.json + pesos), VAE (dir ou arquivo único)

Checks automáticos
- tools/repo_guard.py valida os itens acima
- apps/server/backend/policy.py reforça em runtime

