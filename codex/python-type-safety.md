Python Type Safety (Pyright) – Setup & Usage
===========================================

Why Pyright
- “Absurdly strict” typing that flags drift (e.g., str vs int sliders) without runtime hacks.
- Fast incremental checks; strict null/unknown handling keeps UI glue honest.

Config
- Root config: `pyrightconfig.json` (strict mode; library types enabled).
- Workspace-only mode (default here): focuses on `webui.py`, `launch.py` e módulos `ui*`.
  - Evita ruído quando a venv não está pronta (ex.: ambiente Windows).
  - `reportMissingImports` e `reportMissingTypeStubs` ficam brandos para libs opcionais.
  - Regras de soundness (missing parameter types, overrides incompatíveis, call issues) continuam em erro.

Como rodar (global ou NPX)
- Global (preferido se Pyright estiver instalado):
  - `pyright --stats`
  - `pyright --outputjson > pyright.json`
- Portável (sem instalação global):
  - `npx -y pyright --stats`

Relatórios salvos
- Script utilitário: `scripts/pyright-report.sh`
  - Gera `codex/reports/pyright/pyright-YYYYMMDD-HHMMSS.{txt,json}`.
  - Usa `pyright` global quando disponível e cai para `npx` caso contrário.

Interpretando falhas
- `reportUnknown*`: adicione anotações explícitas (`int | None`, `Annotated[int, …]`, etc.).
- `reportMissingImports`: instale a dependência ou adicione stubs; para deps só de runtime use `# pyright: ignore[reportMissingImports]` na linha.
- `reportMissingTypeStubs`: prefira `pip install types-<package>` ou forneça `*.pyi` em `typings/`.

Política de escopo
- Não relaxe globalmente sem necessidade; mantenha ajustes focados por arquivo/modo.
- Precisa checar backend com venv disponível? Ajuste `include` localmente e promova `reportMissingImports` para erro.
- Prefira ignores direcionados (`# pyright: ignore[rule-id]`) a aberturas gerais.

Próximos passos sugeridos
- Continue tipando hotspots de UI/Gradio (`modules/ui.py`, pipelines de geração). Garanta que payloads de request sejam validados antes do uso.
