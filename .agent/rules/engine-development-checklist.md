---
trigger: always_on
---

# Engine Development Checklist

Pontos críticos ao criar ou modificar engines de difusão. Baseado em lições aprendidas durante o desenvolvimento do Z Image.

Sempre atualize esse arquivo quando surgirem novos detalhes pertinentes.

---

## 1. Memory Management - Smart Offload

### 1.1 Registrar Transformers no Smart Offload

O loader precisa saber quais transformers GGUF/quantizados devem ser carregados em CPU:

```python
# apps/backend/runtime/models/loader.py (~1522)
_SMART_OFFLOAD_TRANSFORMERS = {
    "FluxTransformer2DModel",
    "ZImageTransformer2DModel",
    "ChromaTransformer2DModel",
    "SD3Transformer2DModel",
}
```

**❌ ERRO COMUM:** Esquecer de adicionar um novo transformer → modelo carrega na GPU durante assembly → OOM.

### 1.2 Text Encoders precisam de ModelPatcher

Text encoders externos (GGUF, safetensors) devem ser encapsulados com `ModelPatcher`:

```python
# Wrapper pattern (spec.py)
class ZImageCLIP:
    def __init__(self, text_encoder):
        self.model = text_encoder
        self.patcher = ModelPatcher(
            self.model,
            load_device=memory_management.text_encoder_device(),
            offload_device=memory_management.text_encoder_offload_device(),
        )
```

**Sem isso:**
- Text encoder fica na GPU ocupando memória
- Não é movido corretamente entre CPU/GPU
- OOM ao competir com transformer

### 1.3 Padrão de uso no engine

```python
def get_learned_conditioning(self, prompts):
    # 1. Carregar para GPU via memory manager
    memory_management.load_model_gpu(runtime.clip.patcher)
    unload_clip = self.smart_offload_enabled
    
    try:
        # 2. Processar
        cond = runtime.text.encode(prompts)
        return cond
    finally:
        # 3. Descarregar se smart offload habilitado
        if unload_clip:
            memory_management.unload_model(runtime.clip.patcher)
```

### 1.4 VAE também precisa de load/unload

```python
def decode_first_stage(self, x):
    memory_management.load_model_gpu(self.codex_objects.vae)
    try:
        return runtime.vae.decode(x)
    finally:
        if self.smart_offload_enabled:
            memory_management.unload_model(self.codex_objects.vae)
```

---

## 2. VAE

### 2.1 Lazy Initialization

O VAE usa lazy initialization - `_apply_precision` só é chamado em `decode`/`encode`. 
**Não adicionar conversões ou inicializações durante `__init__`.**

### 2.2 Formato de saída padronizado

O VAE retorna `BCHW` com range `[-1, 1]` diretamente.

**❌ ERRO COMUM:** Adicionar conversões manuais no engine:
```python
# NÃO FAZER - já está no VAE
sample = vae.decode(x).movedim(-1, 1) * 2.0 - 1.0
```

---

## 3. Streaming (Flux-específico)

### 3.1 Suporte a Streaming Core

O Flux tem suporte a streaming para GPUs com pouca VRAM. Se relevante para seu engine:

```python
# spec.py
transformer = _maybe_enable_streaming_core(transformer, spec=spec, engine_options=engine_options)
```

O streaming mantém pesos na CPU e faz upload layer-by-layer durante forward pass.

---

## 4. Condicionamento

### 4.1 Formato esperado pelo sampler

O sampler espera um dict com chaves específicas:

```python
{
    "crossattn": tensor,  # [B, seq_len, hidden_dim] - REQUIRED
    "vector": tensor,     # [B, pooled_dim] - pode ser zeros se não aplicável
    "guidance": tensor,   # [B] float - para distilled CFG
}
```

### 4.2 Distilled CFG

Modelos flow-matching (Flux, Z Image, etc.) usam distilled CFG:

```python
distilled_cfg = getattr(prompts, "distilled_cfg_scale", DEFAULT) or DEFAULT
guidance = torch.full((batch_size,), float(distilled_cfg), dtype=torch.float32)
return {"crossattn": cond, "guidance": guidance}
```

---

## 5. CodexObjects

### 5.1 Retorno de _build_components

O método `_build_components` deve retornar um `CodexObjects`:

```python
return CodexObjects(
    unet=runtime.unet,          # UnetPatcher
    vae=runtime.vae,            # VAE wrapper
    text_encoders={"clip": runtime.clip},  # Dict com wrappers de text encoder
    clipvision=None,            # Opcional
)
```

### 5.2 O `text_encoders` é usado pelo memory manager

As chaves do dict `text_encoders` são usadas para gerenciamento automático. 
O wrapper deve ter `.patcher` acessível.

---

## 6. Engine Options

### 6.1 Passando opções do frontend

Options como `core_streaming_enabled`, `tenc_path` são passadas via `engine_options`:

```python
def _build_components(self, bundle, *, options):
    streaming = options.get("core_streaming_enabled", False)
    tenc_path = options.get("tenc_path")
```

### 6.2 Opções disponíveis no sistema

- `core_streaming_enabled`: Habilita streaming de pesos
- `tenc_path`: Caminho para text encoder externo
- `vae_path`: Caminho para VAE externo

---

## 7. Paridade (Diffusers/ComfyUI) — onde a “golesma” nasce

Quando um engine “funciona” (sem crash) mas só gera ruído, quase sempre é drift em **paridade**:
scheduler/sigmas, semântica de timestep, sinal do modelo, ordem do stream, ou matemática do bloco (norms/residuals).

Links úteis (evidência / leitura aprofundada):
- `.sangoi/reports/2025-12-14-zimage-golesma-parity-postmortem.md`
- `.sangoi/handoffs/HANDOFF_2025-12-14-zimage-golesma-diffusers-parity.md`
- `.sangoi/handoffs/HANDOFF_2025-12-14-zimage-golesma-2-refiner-norm-parity.md`
- `.sangoi/handoffs/HANDOFF_2025-12-14-wan22-14b-scrutiny.md`

### 7.1 Sigma schedule (flow): `shift`, `sigma_min`, tail

Checklist mínimo:
- [ ] Logar o sigma ladder (head + tail) com `CODEX_LOG_SIGMAS=1`.
- [ ] Confirmar `shift` e `sigma_min` batem com os assets HF / implementação de referência.
- [ ] Se o upstream tiver `..., 0, 0` (double-zero tail), isso é esperado (último `dt=0`).

**Anti-padrão:** assumir que “steps=8” ou “shift=1” é universal; para alguns flows (ex. Z-Image Turbo) não é.

### 7.2 Semântica de timestep: `sigma` vs `t_inv`

Alguns pipelines passam `t_norm` como “tempo normalizado” (0 no começo → 1 no final) e o core multiplica por `t_scale`.
Outros passam `sigma` diretamente (1 no começo → 0 no final).

Checklist mínimo:
- [ ] No core, documentar explicitamente qual semântica entra no timestep embedder.
- [ ] Logar nos primeiros steps (debug) os valores derivados: `sigma`, `t_inv`, `t_scaled`.

### 7.3 Sinal (negation): “um `-` a mais” mata tudo

Para modelos flow/velocity, é comum o pipeline precisar negar o output **uma vez** antes da integração.
Se você negar duas vezes (core nega + pipeline nega), o sampler anda na direção errada.

Checklist mínimo:
- [ ] Definir **onde** ocorre a negation (core ou pipeline) e manter só um lugar.
- [ ] Se existir rota “standalone/diffusers-math”, garantir que ela não aplica o `-` duas vezes.

### 7.4 Token stream order + RoPE pos_ids (modelos multimodais)

Para modelos que juntam streams (ex.: image tokens + caption tokens):
- A ordem do `torch.cat` no “unified stream” importa.
- O `pos_ids`/RoPE precisa seguir o upstream (inclui offsets e padding multi-of-32).

Checklist mínimo:
- [ ] Confirmar ordem do stream (ex.: **image→caption** vs **caption→image**).
- [ ] Confirmar `pos_ids` com offsets idênticos ao reference (start coords, padding coords).
- [ ] Confirmar que pads usam `*_pad_token` e que o mask (quando existir) tem semântica correta (bool mask em SDPA: True=keep).

### 7.5 Ordem correta de norms nos blocos *non-modulated* (o “detalhe assassino”)

Quando um bloco roda sem adaLN modulation (ex.: refiner de caption), a ordem correta costuma ser:
- `attn_out = attn(norm1(x))`
- `x = x + norm2(attn_out)`
- `x = x + norm2_ffn(ffn(norm1_ffn(x)))`

**Anti-padrões que causam conditioning drift:**
- Double-norm no input da attention (`norm2(norm1(x))` como entrada).
- `norm2` entrando no MLP (normalização errada no input do FFN).

Esse bug foi a última peça que destravou a saída do Z-Image (de “golesma” para imagens coerentes).

### 7.6 Dtype da integração (flow): latents fp32

Mesmo quando o core roda em bf16/fp16, o integrador/scheduler pode precisar de latents fp32 para evitar drift no tail.

Checklist mínimo:
- [ ] Logar dtype dos latents no loop de sampling.
- [ ] Se o upstream integra em fp32, replicar.

### 7.7 GGUF: `Q4_K_M` não significa “um único quant”

GGUF é quantização **por tensor**. Um arquivo “Q4_K_M” pode misturar:
- tensores BF16 (ex.: embeddings/norms), e
- K-family variados (Q4_K/Q5_K/Q6_K) em diferentes camadas.

Checklist mínimo:
- [ ] Não tratar “apareceu Q5_K/Q6_K no log” como bug automaticamente.
- [ ] Só vira suspeito se: shape/stride inválido, NaNs, ou qtype incompatível com a camada.

---

## Checklist Completo para Novo Engine

### Loader (runtime/models/loader.py)
- [ ] Adicionar transformer ao `_SMART_OFFLOAD_TRANSFORMERS`
- [ ] Verificar se `_load_huggingface_component` trata o novo `cls_name`

### Assembly (spec.py)
- [ ] Criar wrapper CLIP/TextEncoder com `ModelPatcher`
- [ ] Criar estrutura `EngineRuntime` com `clip`, `vae`, `unet`, `text`
- [ ] Retornar `CodexObjects` com todos os componentes

### Engine (engine.py)
- [ ] Implementar `get_learned_conditioning` com load/unload
- [ ] Implementar `encode_first_stage` com load/unload
- [ ] Implementar `decode_first_stage` com load/unload
- [ ] Retornar dict de conditioning com `crossattn` + extras

### Testes
- [ ] Testar com modelos safetensors
- [ ] Testar com modelos GGUF/quantizados
- [ ] Testar com `smart_offload` habilitado
- [ ] Testar em GPU com pouca VRAM
- [ ] Verificar logs de memória (`[memory-debug]`)

---

## Referências

- `apps/backend/engines/flux/` - Engine de referência completo
- `apps/backend/runtime/memory/smart_offload.py` - Funções de offload
- `apps/backend/patchers/base.py` - ModelPatcher
- `apps/backend/patchers/vae.py` - VAE wrapper
- `.sangoi/templates/wan22-gguf-request.json` — payload de referência para WAN22 GGUF (high/low + assets).
- `.sangoi/reports/comparisons/wan22-img2vid-mapping.md` — mapping de inputs/canais para WAN22 I2V.
- `.sangoi/research/runtime/wan22-text-encoder-compat.md` — TE configs/tokenizer (vendored) vs pesos (user-supplied).
