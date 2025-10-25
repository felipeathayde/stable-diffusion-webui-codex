Backend Services Overview
=========================

Project Identity
- This repository is `stable-diffusion-webui-codex`. The goal is a native backend (`apps/server/backend`) with zero runtime dependence on A1111/Forge. The `legacy/` folder remains read‑only for reference only.
- Maintenance uses OpenAI “codex” as a coding assistant in the development workflow; there is no LLM integrated into the runtime.
- The `legacy/` folder contains a snapshot of Forge’s `main` to serve as a functional pipeline reference during refactors.

Purpose
- Reduce duplication and decouple FastAPI routes from generation internals.
- Create a thin service layer that centralizes common flows and helpers.

Services (native)
- ImageService (backend/services/image_service.py)
  - Orquestra geração a partir dos engines/tarefas.
  - Reporta progresso via `core/state.py` e publica eventos para os serviços.

- MediaService (backend/services/media_service.py)
  - decode_image(str): base64 ou URL (guardas por env: `CODEX_API_*`).
  - encode_image(PIL.Image): respeita `CODEX_SAMPLES_FORMAT`, `CODEX_JPEG_QUALITY`, `CODEX_WEBP_LOSSLESS`.

- OptionsService (backend/services/options_service.py)
  - get/set em `apps/settings_values.json` (sem sysinfo do legado).
  - Flags de linha de comando: não exposto; usar env `CODEX_*` quando necessário.

- SamplerService (backend/services/sampler_service.py)
  - resolve(name, scheduler): normaliza nomes via `SamplerKind`.
  - ensure_valid_sampler(name): 400 se inválido.

API Changes (modules/api/api.py)
- API constructs p_factory/prepare_p and delegates to ImageService.
- All image encode/decode moved to MediaService.
- Options and flags routes delegate to OptionsService.
- Sampler resolution/validation uses SamplerService.

Behaviour
- Erros explícitos e crash dumps em `logs/exceptions-YYYYmmdd-<pid>.log`.
- Sem fallbacks silenciosos para legado; se faltou funcionalidade ainda não portada, `NotImplementedError`.

Next
- Optional: ProgressService to centralize task/time metrics and future telemetry.
- Optional: move sampler/scheduler listing to SamplerService.

Tokenizer Assets (Strict + Online Cache)
-------------------------

- Tokenizers are not embedded in `.safetensors`. They must exist in the model directory (or be cached locally from the Hub) as standard files:
  - Preferred: `tokenizer.json` (+ `tokenizer_config.json`)
  - BPE: `vocab.json` + `merges.txt`
  - SentencePiece: `tokenizer.model`
- SDXL requires two tokenizer folders: `tokenizer/` and `tokenizer_2/`.
- Loader behavior (backend/loader.py):
  - Primeiro tenta resolver no disco (pasta do componente e raiz do repositório local).
  - Se faltar e `--disable-online-tokenizer` NÃO estiver habilitado e `huggingface_repo` for conhecido, baixa do Hub apenas os artefatos do tokenizer e copia para o diretório do modelo (cache explícito, sem “magia”).
  - Se ainda assim não existir ou falhar, erro explícito (sem mascarar). Não há fallback silencioso.
- Recommended workflow when you only have a checkpoint `.safetensors`:
  - Download tokenizers (and `config.json`) from the original repo (e.g., via `huggingface_hub.snapshot_download(allow_patterns=["tokenizer.*", "vocab.json", "merges.txt", "tokenizer.model", "tokenizer_config.json", "config.json"])`).
  - Place them under the model directory alongside your weights.
