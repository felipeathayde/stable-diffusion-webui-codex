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
