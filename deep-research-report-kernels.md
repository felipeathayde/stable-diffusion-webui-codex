# Kernels low-level inspirados em FlashAttention no repositório stable-diffusion-webui-codex

## Resumo executivo

Este repositório centraliza múltiplos “engines” de difusão (UNet clássico SD1/2/SDXL e transformadores modernos como Flux/Chroma/Z-Image/WAN22) e, por consequência, executa **atenção** em vários formatos (self-attn/cross-attn; 2D e 3D; com RoPE; com chunking). A boa notícia é que a base já está preparada: existe um **dispatcher de backends de atenção** (PyTorch SDPA, xFormers, split/chunked) e o engine WAN22 já expõe seleção explícita de backend SDPA (“flash”, “mem_efficient”, “math”) + chunking em nível de wrapper. A oportunidade é transformar isso em uma estratégia unificada e “FlashAttention-first” nos hotpaths, com fallback robusto.

O maior potencial prático de ganhos (memória + latência + throughput) está em três núcleos:  
1) **Dispatcher de atenção do runtime** (`apps/backend/runtime/attention/__init__.py`), que hoje escolhe entre PyTorch SDPA / xFormers / split / sub-quadratic. É o ponto de substituição mais direto para inserir um backend “flash-attn” (via biblioteca FlashAttention) e/ou um modo “forçar SDPA flash” (via `sdpa_kernel`) com fallback.  
2) **Transformers de difusão com sequências longas**: Flux e Chroma usam atenção em blocos com qkv projetado e RoPE, chamando o dispatcher central (ótimo para plugar kernels) e, portanto, são candidatos prioritários para FlashAttention-style tiling.  
3) **Z-Image e WAN22**: ambos implementam SDPA diretamente em módulos próprios; WAN22 já tem wrapper SDPA com seleção de backend e chunking, mas pode se beneficiar de um backend “flash-attn” dedicado (ou do aproveitamento consistente do backend FLASH do PyTorch em todo o projeto). FlashAttention foi desenhada exatamente para reduzir I/O entre HBM e SRAM on-chip via tiling e softmax “online”, mantendo exatidão e reduzindo footprint de memória (linear em vez de quadrático em muitos cenários). citeturn30search0turn33view0

Recomendação de alto nível (ordem de impacto/risco):  
- **Camada 0 (baixo risco, alto ROI)**: unificar e expor controle de backend SDPA (flash/mem_efficient/math) no dispatcher central, copiando a disciplina do WAN22 (`torch.nn.attention.sdpa_kernel`). citeturn30search2turn33view0  
- **Camada 1 (alto impacto, risco moderado)**: integrar a biblioteca FlashAttention (flash-attn) como backend opcional do dispatcher para os casos de sequências grandes e dtypes fp16/bf16, com fallback. citeturn33view0turn32search1  
- **Camada 2 (engenharia avançada)**: adicionar kernels Triton (ex.: fused attention do tutorial oficial) para cobrir lacunas de compatibilidade/feature (máscaras específicas, formatos, etc.) e como caminho “mais portável” dentro de Linux/WSL. citeturn32search5turn33view0

## Fundamentos do FlashAttention

FlashAttention (e evoluções como FlashAttention-2/3) é uma família de algoritmos/kernels para atenção exata cujo foco é **I/O-awareness**: o gargalo em atenção não é apenas FLOPs; frequentemente é **movimentação de dados** (ler/escrever grandes matrizes intermediárias em HBM). O mecanismo clássico computa `S = QKᵀ`, depois `P = softmax(S)` e por fim `O = PV`; isso tende a **materializar** `S` e/ou `P` (matrizes N×N) e realizar leituras/escritas caras em HBM. FlashAttention evita isso ao fazer **tiling por blocos** e aplicar uma softmax “online” (mantendo running max/sum por bloco) mantendo intermediários em registradores e **shared memory (SRAM on-chip)**, reduzindo drasticamente tráfego HBM. citeturn30search0turn33view0turn32search4

### Por que blocos + SRAM/shared memory ajudam

Em GPUs CUDA, a hierarquia de memória inclui global (HBM), caches e **shared memory por bloco**, que é on-chip, baixa latência e alta largura de banda para threads do mesmo bloco. A ideia do tiling é carregar “tiles” de K/V (e às vezes Q) para shared memory e reutilizá-los, evitando leituras repetidas da global e evitando escrever intermediários grandes. A própria documentação da NVIDIA descreve shared memory como espaço visível ao bloco, com ciclo de vida do bloco e papel central em otimizações por tiling. citeturn32search4turn32search2

FlashAttention-2, além do princípio base, enfatiza **particionamento de trabalho** mais eficiente entre thread blocks e warps e redução de comunicação via shared memory, aproximando eficiência de GEMM em GPUs como A100/H100 (em benchmarks de treinamento). citeturn31search0turn33view0  
FlashAttention-3 mira especificamente Hopper (H100/H800) e impõe requisitos de CUDA e GPU (p. ex. CUDA ≥ 12.3). citeturn33view0

### SDPA no PyTorch e a relação com FlashAttention

O projeto já usa amplamente `torch.nn.functional.scaled_dot_product_attention` (SDPA). O PyTorch oferece um context manager (`torch.nn.attention.sdpa_kernel`) para **selecionar o backend** (Flash, Efficient, Math). Isso é crucial porque dá uma via de “quase FlashAttention” sem dependências extras, quando o backend FLASH está disponível e compatível com o caso. citeturn30search2

## Mapeamento do repositório e hotspots candidatos a kernels FlashAttention-style

Abaixo estão os pontos mais “carregados” (atenção/softmax/matmul/transposes) que aparecem como **candidatos naturais** a substituição/otimização. As referências de “linha aproximada” são indicativas (a posição exata pode variar com commits), mas o arquivo e o símbolo são canônicos.

### Dispatcher central de atenção do runtime

**Arquivo:** `apps/backend/runtime/attention/__init__.py`  
**Símbolos principais:** `attention_basic`, `attention_split`, `attention_sub_quad`, `attention_xformers`, `attention_pytorch`, `attention_function`, `AttentionProcessorCodex`.

**O que acontece aqui (hot ops):**
- `attention_basic`: projeta Q/K/V em layout por heads, computa `sim = einsum(q, k)`, aplica máscara, faz `softmax`, e então `out = einsum(sim, v)`. Isso envolve `einsum` (matmul), `softmax` e reshape/permute (transposição).  
- `attention_split`: calcula por fatias de queries em loop Python; repete `einsum -> softmax -> einsum`. Reduz pico de memória, mas aumenta overhead e perde fusão.  
- `attention_sub_quad`: chama `efficient_dot_product_attention` (sub-quadrático) com chunking; usa `baddbmm/bmm`, softmax e lógica de chunk baseada em memória livre.  
- `attention_pytorch`: usa diretamente `scaled_dot_product_attention` e faz `transpose/reshape` para voltar ao formato [B, tokens, dim].  
- `attention_function`: dispatcher por backend via config de memória (XFORMERS/PYTORCH/SPLIT/QUAD).  

**Candidato FlashAttention-style:** inserir um novo backend, por exemplo `FLASH_ATTN` (biblioteca flash-attn) e/ou `PYTORCH_FLASH_FORCED` (SDPA com `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`), com fallback para `attention_pytorch`/`attention_xformers`.

Por que é o melhor “ponto de alavanca”: já é a função chamada pelo UNet clássico e por Flux/Chroma via `attention_function`, então um único kernel novo pode beneficiar múltiplas famílias sem “cirurgia” ampla.

### UNet clássico (SD1/2/SDXL) e seus SpatialTransformers

**Arquivo:** `apps/backend/runtime/common/nn/unet/layers.py`  
**Símbolos:** `CrossAttention`, `BasicTransformerBlock`, `SpatialTransformer`.

**Operações relevantes:**
- `CrossAttention.forward`: computa `q = to_q(x)`, `k = to_k(context)`, `v = to_v(context)` e chama `attention_function(q,k,v, heads, mask)`. Aqui há **três GEMMs (lineares)** + atenção.  
- `BasicTransformerBlock._forward`: contém caminhos de “patch replace” que podem chamar funções customizadas em `attn1`/`attn2`, inclusive gerando Q/K/V explicitamente e chamando patch. Isso é importante porque uma integração FlashAttention deve respeitar esses hooks.  
- `SpatialTransformer.forward`: rearranja `b c h w -> b (h*w) c` (flatten espacial) e depois `b (h*w) c -> b c h w`. Essa transposição/contiguidade pode impactar performance e é um local para evitar cópias desnecessárias quando possível.

**Candidato FlashAttention-style:** substituir o miolo de `attention_function` para que o UNet use um kernel FlashAttention (fusão `QKᵀ + softmax + PV`) sem materializar scores em HBM.

### Flux e Chroma (transformers com RoPE; sequências potencialmente longas)

**Arquivos-chave:**
- `apps/backend/runtime/families/flux/components.py` (classes `SelfAttention`, `DoubleStreamBlock`, `SingleStreamBlock`)  
- `apps/backend/runtime/families/flux/flux.py` (função `attention(q,k,v,pe)` que chama `attention_function(..., skip_reshape=True)`)  
- `apps/backend/runtime/families/chroma/chroma.py` (blocagem muito similar; usa `attention_function(..., skip_reshape=True)`)

**Operações relevantes:**
- QKV via `nn.Linear` gerando `[B, seq, 3*dim]`, reshape para `[B, heads, seq, head_dim]`, aplicação de RoPE, concatenação de sequências (no joint attention), e finalmente SDPA via dispatcher.  
- Há muitas `view`, `permute`, `torch.cat`, e o SDPA no centro.

**Por que é um hotspot ideal para FlashAttention:** Em modelos DiT/flow-transformers, a sequência de tokens cresce com a resolução (patchificação). O custo quadrático de atenção domina rapidamente e é precisamente o cenário onde FlashAttention (tiling + SRAM) tende a trazer ganhos de memória e throughput quando comparado a implementações que materializam intermediários. citeturn30search0turn33view0

### Z-Image (atenção própria usando SDPA direto)

**Arquivo:** `apps/backend/runtime/families/zimage/model.py`  
**Símbolo:** classe `Attention` (atenção com QKV combinado + RoPE + SDPA).

**Operações relevantes:**
- `qkv = Linear(x)` e reshape para `[B, N, 3, H, D]`, split Q/K/V.  
- RoPE via representação complexa e `view_as_complex/view_as_real` (custo relevante por token).  
- Transpose para `[B, H, N, D]` e chamada a `F.scaled_dot_product_attention`.  

**Candidato FlashAttention-style:** dois caminhos:
1) Roteá-lo para o dispatcher `attention_function` (ganha unificação de backend e aproveita FlashAttention nas mesmas regras).  
2) Implementar chamada direta à flash-attn (qkvpacked), porque Z-Image já produz QKV “fused” no formato natural para isso.

### WAN22 (vídeo 3D; SDPA com política e chunking já existente)

**Arquivos-chave:**
- `apps/backend/runtime/families/wan22/model.py` (`WanSelfAttention`, `WanCrossAttention`)  
- `apps/backend/runtime/families/wan22/sdpa.py` (wrapper `sdpa(...)` com seleção de backend SDPA e chunking)

**Operações relevantes e já “Flash-aware”:**
- `WanSelfAttention` e `WanCrossAttention` fazem Q/K/V projections, RoPE, rearranjos e chamam `wan_sdpa(q,k,v)`.  
- `wan22/sdpa.py` usa `torch.nn.attention.sdpa_kernel` (quando disponível) para escolher backend (flash/mem_efficient/math) e oferece modos `global` e `sliding` com `chunk` em loop (atenção por janela/sliding ou chunking global). Isso é um mecanismo explícito para reduzir memória e controlar performance — muito alinhado com a motivação do FlashAttention. citeturn30search2

**Candidato FlashAttention-style:** expandir o wrapper para:
- suportar um backend flash-attn dedicado quando o PyTorch FLASH não cobrir um caso (ou quando flash-attn for mais eficiente),  
- ou levar esse mesmo mecanismo (`sdpa_kernel` + chunk) para o dispatcher central do repositório para consistência entre engines.

### Atenção sub-quadrática (malloc/softmax/loops) como fallback e alvo secundário

**Arquivo:** `apps/backend/runtime/misc/sub_quadratic_attention.py`  
Implementa `efficient_dot_product_attention` com chunking e softmax “online” parcial. É um fallback útil para reduzir pico, mas ainda executa várias primárias (`baddbmm`, `softmax`, `bmm`) e loops em Python. É um alvo secundário para troca por kernel fused (FlashAttention-like) quando a prioridade for **latência** e não apenas “não OOM”.

## Candidatos priorizados com ganhos esperados, complexidade, dependências e riscos

A seguir, uma visão analítica por “ponto de substituição”, com ênfase no que muda ao adotar kernels FlashAttention-style.

### Backend FlashAttention no dispatcher central

**Onde:** `apps/backend/runtime/attention/__init__.py` → `attention_function(...)` (direcionador).  
**O que substituir:** adicionar `attention_flash_attn(...)` e plugar no dispatcher.

**Ganho esperado**
- **Memória:** tende a reduzir pico, especialmente para sequências grandes (atenção quadrática), ao evitar materialização de scores/probs em HBM e usar softmax online/tiling. citeturn30search0turn33view0  
- **Latência/throughput:** ganho variável; em geral mais relevante quando a atenção é memory-bound e/ou a sequência é grande. FlashAttention-2 mira eficiência próxima de GEMM em GPUs modernas, melhorando ocupação/particionamento. citeturn31search0turn33view0

**Complexidade**
- Média: integração em Python + gate por shape/dtype + fallback.
- Alta se você for escrever kernel do zero; baixa/média se usar a biblioteca flash-attn.

**Dependências**
- Opção A: apenas PyTorch (usar SDPA + `sdpa_kernel`). citeturn30search2  
- Opção B: `flash-attn` (CUDA extension; compilação). citeturn33view0turn32search1  
- Opcional: Triton para fallback custom (tutorial oficial tem “Fused Attention”). citeturn32search5

**Riscos**
- Compatibilidade com máscaras (boolean/additive), formatos e dtypes.
- Diferenças numéricas pequenas (tolerâncias) — inevitável ao mudar kernel/ordem de redução.
- Interação com hooks de patch/LoRA no UNet: não pode “engolir” a semântica de `patches_replace` e as rotas que já alteram Q/K/V.

**Windows**
- A biblioteca flash-attn declara Linux como requisito principal; há indicação de que pode funcionar em Windows a partir de certa versão, mas ainda exige mais testes/infra de wheels. citeturn33view0  
- Em prática, para Windows, o caminho mais previsível costuma ser WSL2 (Linux) ou ficar no SDPA do PyTorch.

### Forçar SDPA FLASH como “modo de baixo risco”

**Onde:** também no dispatcher central, mas mudando a implementação de `attention_pytorch(...)` para opcionalmente envolver SDPA em `sdpa_kernel(...)` com backend FLASH (e fallback para efficient/math).  
**Ganho esperado**
- Pode destravar o backend FlashAttention do PyTorch quando ele está disponível e compatível, sem dependências extras. citeturn30search2

**Complexidade**
- Baixa.
- Alto valor porque já existe precedente no `wan22/sdpa.py` usando essa abordagem.

**Riscos**
- Se você “desabilitar” backends demais, certos formatos podem falhar. A solução é: tentar FLASH, e se falhar, refazer com Efficient/Math.

### Fusão QKV + atenção nos transformers Flux/Chroma/Z-Image

**Onde:**
- Flux/Chroma: blocos em `apps/backend/runtime/families/flux/components.py` e `apps/backend/runtime/families/chroma/chroma.py`.  
- Z-Image: `apps/backend/runtime/families/zimage/model.py` (QKV já fused).

**Ideia**
- Quando QKV já está “empacotado” (Linear gera 3*dim), você pode preferir rotas “qkvpacked” do flash-attn, reduzindo overhead de rearranjos e permitindo kernels mais especializados.

**Ganho esperado**
- **Latência:** melhor por reduzir transposes/contiguous e por aproveitar kernel otimizado no formato nativo.
- **Memória:** melhora por reduzir intermediários grandes.

**Complexidade**
- Média: envolve ajustar chamadas (e talvez formatos) mas não reescrever modelos inteiros.

**Riscos**
- Layouts (B,H,N,D vs B,N,H,D) e RoPE: você deve aplicar RoPE antes no layout certo, sem cópias excessivas.
- Em Flux/Chroma há “joint attention” (concatenação txt+img); o backend deve suportar isso.

### WAN22: substituir o SDPA wrapper por flash-attn quando aplicável

**Onde:** `apps/backend/runtime/families/wan22/sdpa.py` e chamadas em `WanSelfAttention/WanCrossAttention`.  
**Estado atual**
- Já permite escolher backend SDPA (flash/mem_efficient/math) e usar chunking/sliding. citeturn30search2

**Extensão FlashAttention-style**
- Inserir um caminho “flash-attn library” quando `policy == flash` e `q.is_cuda` e condições de dtype/head_dim/causal forem compatíveis.
- Se não for compatível, cair de volta para SDPA do PyTorch (como já faz), possivelmente mantendo chunking.

**Risco/compatibilidade**
- Para vídeo, seq_len pode ficar grande; chunking/sliding vai continuar útil em casos extremos, mesmo com flash-attn.

## Design de kernel e pseudocódigo

Esta seção descreve o desenho conceitual (inspirado nos papers) e como isso se traduz para kernels em CUDA/Triton. O objetivo é: **fusão + tiling + softmax online**.

### Esqueleto do kernel FlashAttention-style

**Problema:** `O = softmax(QKᵀ) V` sem materializar `QKᵀ` nem `softmax(QKᵀ)` em HBM.

**Estratégia (resumo)**
- Dividir Q em blocos (tile de queries) e K/V em blocos (tile de keys/values).
- Para cada bloco de Q, iterar blocos de K/V:
  - Computar bloco de scores `S_block = Q_block @ K_blockᵀ` (em registradores / shared).
  - Atualizar estatísticas de softmax de forma estável: max e soma acumulada (online).
  - Acumular `O_block += P_block @ V_block` sem armazenar `P_block`.

Isso reduz acessos à HBM e explora shared memory (SRAM on-chip). citeturn30search0turn32search4turn32search2

### Pseudocódigo de alto nível (forward)

```text
inputs: Q[B,H,M,D], K[B,H,N,D], V[B,H,N,D]
output: O[B,H,M,D]
params: tile_m, tile_n, scale = 1/sqrt(D)

for each (b, h):
  for m0 in range(0, M, tile_m):
    Qm = load Q[b,h,m0:m0+tile_m,:]            // registers
    m_i = -inf[ tile_m ]                       // running max per query
    l_i = 0[ tile_m ]                          // running sum(exp)
    Oi  = 0[ tile_m, D ]                       // running output accumulator

    for n0 in range(0, N, tile_n):
      Kn = load K[b,h,n0:n0+tile_n,:] into shared memory
      Vn = load V[b,h,n0:n0+tile_n,:] into shared memory

      S  = (Qm @ Kn^T) * scale                 // tile_m x tile_n
      if mask: S += mask[m0:m0+tile_m, n0:n0+tile_n]

      m_new = max(m_i, rowmax(S))
      P = exp(S - m_new[:,None])
      l_new = l_i * exp(m_i - m_new) + rowsum(P)

      Oi = Oi * exp(m_i - m_new)[:,None] + (P @ Vn)
      m_i = m_new
      l_i = l_new

    O[b,h,m0:m0+tile_m,:] = Oi / l_i[:,None]
```

**Notas de implementação**
- Em CUDA moderno, `Qm @ Knᵀ` e `P @ Vn` podem ser implementados via Tensor Cores usando WMMA/MMA, e o tile sizes dependem de head_dim/D e da arquitetura (capacidade de shared memory e registros). citeturn32search4turn32search2  
- A estabilidade numérica vem do par (`m_i`, `l_i`) e da reescala `exp(m_i - m_new)`.

### Heurísticas de tile e variáveis a considerar

Como o usuário não especificou GPU/CUDA/PyTorch, estes fatores devem ser tratados como variáveis de projeto:

- **Arquitetura da GPU (SM, shared mem/SM, Tensor Cores)**: influencia tile sizes e se vale usar kernels específicos (ex.: FlashAttention-3 em Hopper). citeturn33view0turn32search4  
- **dtype (fp16/bf16/fp32)**: FlashAttention-2 em CUDA foca fp16/bf16; bf16 depende de GPUs modernas. citeturn33view0  
- **head_dim**: muitos kernels têm faixas de suporte (p.ex. até 256 em FlashAttention-2). citeturn33view0  
- **tamanho da sequência (N)**: quanto maior N, mais o ganho tende a ser limitado por I/O em HBM (onde FlashAttention brilha). citeturn30search0  
- **máscaras e modo**: causal vs não-causal; “sliding window” vs global; atenção com bias (ALiBi etc.).  
- **layout de tensores**: B×H×N×D vs B×N×H×D e necessidade de `contiguous()`.

### Triton como alternativa de kernel (mais rápido para prototipar)

A documentação oficial do Triton inclui tutorial de “Fused Attention”, além de fused softmax e matmul, sendo um caminho de prototipagem para kernels de atenção por blocos. citeturn32search5

## Plano de integração na WebUI

A meta aqui é introduzir kernels de atenção mais rápidos **sem quebrar** engines, LoRAs, patchers e builds (especialmente em ambientes Windows/WSL).

### Arquitetura de integração

```mermaid
flowchart TD
  A[Config de runtime\nAttentionBackend + policy] --> B[attention_function\nDispatcher]
  B -->|xformers| C[xFormers attention]
  B -->|pytorch| D[PyTorch SDPA\nscaled_dot_product_attention]
  B -->|flash-attn opcional| E[FlashAttention (flash-attn)\nCUDA extension]
  B -->|split/quad| F[Chunked/Sub-quad fallback]
  B --> G[Model families\nUNet / Flux / Chroma]
  G --> B
  H[WAN22 sdpa wrapper] -->|hoje| D
  H -->|extensão| E
```

### Passos recomendados

**Passo de base**
1) **Unificar o conceito “SDPA policy” no runtime**: trazer para o dispatcher central algo equivalente ao `wan22/sdpa.py` (política: flash/mem_efficient/math; modo: global/sliding; chunk). O `sdpa_kernel` do PyTorch é o mecanismo oficial para selecionar backend. citeturn30search2  
2) **Adicionar fallback programático**: tentar FLASH; se falhar por compatibilidade, cair para Efficient/Math automaticamente.

**Integração flash-attn (biblioteca)**
3) Adicionar um backend `FLASH_ATTN` no enum/config do projeto e uma checagem de disponibilidade (`import flash_attn`) com fallback.  
4) Implementar uma função `attention_flash_attn(...)` dentro de `apps/backend/runtime/attention/__init__.py` e conectá-la em `attention_function`.  
5) Gate por critérios (exemplos práticos):
   - `q.is_cuda == True`
   - dtype fp16/bf16
   - formatos suportados (B,H,N,D)
   - head_dim em faixa suportada
6) **Windows/WSL**: tratar flash-attn como dependência opcional; o upstream do flash-attn recomenda Linux e menciona que Windows “pode funcionar” a partir de certa versão, mas ainda requer testes e infraestrutura de wheels. Isso deve ser refletido em documentação e fallback. citeturn33view0

**Harmonizar modelos que usam SDPA direto**
7) Z-Image: decidir se
   - roteia para `attention_function` (centraliza backends), ou
   - chama flash-attn diretamente (qkvpacked) para máxima performance.  
8) WAN22: opcionalmente estender `wan22/sdpa.py` para aceitar `flash-attn` como backend alternativo quando `policy=flash` (mantendo chunking/sliding para casos extremos).

### Testes e “fallback-first” (sem regressão)

9) Criar testes de correção por engine:
   - Comparar saída do modelo (ou atenção isolada) entre backends: PyTorch SDPA vs flash-attn vs xformers, com tolerâncias. O próprio flash-attn testa equivalência numérica com tolerância vs baseline PyTorch. citeturn33view0  
10) Criar um modo “safe”:
   - Se qualquer falha de import/compatibilidade ocorrer, retornar **automaticamente** ao backend PYTORCH (ou ao backend anterior) e logar claramente.

## Alternativas, testes e referências

### Tabela comparativa de alternativas

| Alternativa | Onde encaixa melhor | Prós | Contras | Windows |
|---|---|---|---|---|
| FlashAttention (flash-attn, CUDA ext) | Hotpaths de atenção com seq grande (Flux/Chroma/UNet/Z-Image/WAN22) | Kernel altamente otimizado; foco explícito em reduzir I/O HBM↔SRAM; inclui FA2/FA3; suporte a fp16/bf16 e head_dim amplos (conforme docs do projeto) citeturn33view0turn32search1 | Exige toolchain CUDA e compilação; compatibilidade depende de GPU/dtype/shape; FA3 requer Hopper+CUDA>=12.3 citeturn33view0 | Linux é foco; Windows “pode funcionar” mas requer mais testes citeturn33view0 |
| PyTorch SDPA (backends selecionáveis via `sdpa_kernel`) | Caminho de menor fricção; já usado no repo | Sem dependência extra; API oficial para forçar backend (FLASH/EFFICIENT/MATH) citeturn30search2 | Nem sempre o backend FLASH é aplicável; fallback pode ocorrer por constraints | Depende do build do PyTorch; geralmente mais amigável |
| xFormers attention | Já integrado no dispatcher central | Boa performance em alguns cenários; alternativa madura em comunidades diffusion | Dependência extra; variações de compatibilidade; debug mais difícil | Pode ser difícil no ecossistema Windows puro |
| Triton (fused attention) | Protótipo/“cobertura” de casos não atendidos por flash-attn | Desenvolvimento mais rápido que CUDA puro; tutorial oficial para fused attention citeturn32search5 | Performance pode ficar abaixo de kernels CUDA hiper-otimizados; compatibilidade depende de stack | Em geral mais previsível via Linux/WSL |
| cuDNN/cuBLAS/cuBLASLt (via PyTorch) | Matmuls e kernels internos do framework | Extremamente otimizados para GEMM; PyTorch já os usa | Você tem pouco controle direto fora das APIs do framework; atenção fused é mais específica | N/A (indireto via PyTorch) |
| JAX/XLA | Se existisse pipeline em JAX | Compilação agressiva e fusões via XLA | Não se integra diretamente ao runtime PyTorch do repo sem reescrita | Fora de escopo operacional (neste repo) |

### Checklist de testes de performance e correção

**Correção**
- Comparar `attention_function` (saída) entre backends (pytorch vs flash-attn vs xformers) para tensores aleatórios com seeds fixos.
- Cobrir casos:
  - self-attn e cross-attn
  - com/sem máscara
  - causal e não-causal (WAN22 “sliding/global”)
  - diferentes head_dim (64/128/256) e seq_len (curto/médio/alto)
- Validar tolerâncias (erro relativo/absoluto) e estabilidade (sem NaNs/Infs).

**Performance**
- Medir: latência por step e por forward do core (UNet/Transformer), tokens/s, VRAM peak, bandwidth.
- Ferramentas: `torch.profiler` e logs do runtime.
- Cenários:
  - resolução baixa vs alta (onde seq_len cresce para DiT/patch)
  - batch size variando
  - fp16 vs bf16
- Acompanhar a efetividade do backend (logar se está “flash” ou caiu para efficient/math). Isso já é feito no WAN22 sdpa wrapper; é uma boa prática para replicar no dispatcher central.

### Referências priorizadas

- Repositório oficial FlashAttention (implementação, requisitos, FA2/FA3, instalação, notas de Windows). citeturn33view0turn32search1  
- Context manager `torch.nn.attention.sdpa_kernel` (seleção de backend SDPA no PyTorch). citeturn30search2  
- Hierarquia de memória CUDA (global/shared/const/texture) e princípios de uso. citeturn32search4  
- Uso/configuração de shared memory (NVIDIA Technical Blog). citeturn32search2  
- Tutoriais do Triton (inclui “Fused Attention” como referência de kernel-by-tiles). citeturn32search5  
- Visão geral do problema e motivação IO-aware (resumo técnico do paper FlashAttention). citeturn30search0