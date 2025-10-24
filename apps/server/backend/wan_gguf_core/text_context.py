from __future__ import annotations

from typing import Optional, Tuple


def _as_dtype(dtype: str):
    import torch
    return {
        "bf16": getattr(torch, "bfloat16", torch.float16),
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.float16)


def encode_with_text_encoder(model_dir: str, prompt: str, negative: Optional[str], *, device: str, dtype: str, text_encoder_dir: Optional[str] = None, tokenizer_dir: Optional[str] = None) -> Tuple[object, object]:
    """Encode prompts using transformers tokenizer + encoder from the model_dir/text_encoder.

    Tries UMT5EncoderModel first, then T5EncoderModel.
    Returns (prompt_embeds, negative_embeds) as torch tensors on `device` with `dtype`.
    """
    import torch
    from transformers import AutoTokenizer  # type: ignore
    from transformers import AutoConfig  # type: ignore
    try:
        from transformers import UMT5EncoderModel as _Enc  # type: ignore
    except Exception:
        from transformers import T5EncoderModel as _Enc  # type: ignore

    import os
    # Build candidate roots: prefer explicit paths, then model_dir
    tk_dir = tokenizer_dir
    te_dir = text_encoder_dir
    te_file: Optional[str] = None
    # If a file path (.safetensors) was provided for the encoder, remember it
    if te_dir and os.path.isfile(te_dir) and te_dir.lower().endswith('.safetensors'):
        te_file = te_dir
        te_dir = os.path.dirname(te_dir)
    if tk_dir and os.path.isfile(tk_dir):
        tk_dir = os.path.dirname(tk_dir)

    # Try a few candidates for tokenizer/encoder resolution
    tok: Optional[object] = None
    enc: Optional[object] = None
    # Also consider local cache under models/tokenizers/<name>
    tok_cache = os.path.join('models', 'tokenizers', 'Wan2.2-I2V-A14B-Diffusers')
    tok_candidates = [p for p in [tk_dir, tok_cache, os.path.join(model_dir, 'tokenizer'), model_dir] if p]
    enc_candidates = [p for p in [te_dir, os.path.join(model_dir, 'text_encoder'), model_dir] if p]
    last_err: Optional[Exception] = None
    for cand in tok_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand, subfolder=None, use_fast=True, local_files_only=True)
            break
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            tok = None
    if tok is None and last_err is not None:
        raise last_err
    # Encoder resolution:
    # 1) If a direct safetensors file was provided, construct from config and load state_dict from file.
    if te_file is not None:
        try:
            cfg = None
            # Try local config.json first
            for cand in enc_candidates:
                try:
                    cfg = AutoConfig.from_pretrained(cand, subfolder=None, local_files_only=True)
                    break
                except Exception:
                    continue
            if cfg is None:
                # Fallback to preset repo small config (download-only config)
                try:
                    cfg = AutoConfig.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="text_encoder", local_files_only=False)
                except Exception as ex:
                    raise RuntimeError(f"failed to resolve text encoder config near {te_dir}: {ex}")
            enc = _Enc(cfg)  # type: ignore[call-arg]
            # Load state dict from the provided safetensors file
            from safetensors.torch import load_file as _load_st
            sd = _load_st(te_file)
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            if len(missing) > 0:
                # Not fatal: just log via print (no logger here)
                print(f"[wan-gguf-core] text encoder missing keys: {len(missing)}; unexpected: {len(unexpected)}")
        except Exception as ex:
            last_err = ex
            enc = None
    # 2) Else try directory-based from_pretrained locally
    if enc is None:
        for cand in enc_candidates:
            try:
                enc = _Enc.from_pretrained(cand, subfolder=None, torch_dtype=_as_dtype(dtype), local_files_only=True)
                break
            except Exception as ex:  # noqa: BLE001
                last_err = ex
                enc = None
    # 3) If still None, bubble the last error
    if enc is None and last_err is not None:
        raise last_err

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    enc = enc.to(dev)
    try:
        enc = enc.to(dtype=_as_dtype(dtype))  # downcast if supported
    except Exception:
        pass

    def _do(txt: str):
        inputs = tok([txt], padding="max_length", truncation=True, max_length=225, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = enc(**inputs).last_hidden_state  # [B, L, C]
            return out.to(_as_dtype(dtype))

    p = _do(prompt or "")
    n = _do(negative or "") if negative is not None else _do("")
    return p, n


def encode_with_pipeline(model_dir: str, prompt: str, negative: Optional[str], *, device: str, dtype: str, vae_dir: Optional[str] = None) -> Tuple[object, object]:
    """Fallback to Diffusers WanPipeline.encode_prompt ensuring dims match UNet.
    Loads only what is necessary.
    """
    import torch
    from diffusers import WanPipeline  # type: ignore
    from diffusers import AutoencoderKLWan  # type: ignore

    torch_dtype = _as_dtype(dtype)
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    # Load minimal VAE to satisfy pipeline init (some versions may require it)
    try:
        import os
        vd = vae_dir
        if vd and os.path.isfile(vd):
            vd = os.path.dirname(vd)
        vae = AutoencoderKLWan.from_pretrained(vd or model_dir, subfolder=(None if vd else "vae"), torch_dtype=torch_dtype, local_files_only=True)
    except Exception:
        vae = None

    pipe = WanPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype, local_files_only=True, vae=vae)
    pipe = pipe.to(dev)
    out = pipe.encode_prompt(
        prompt=prompt or "",
        negative_prompt=negative or "",
        do_classifier_free_guidance=True,
        device=dev,
        dtype=torch_dtype,
    )
    # Try to unpack common return structures
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], out[1]
    if hasattr(out, "prompt_embeds"):
        return out.prompt_embeds, getattr(out, "negative_prompt_embeds", out.prompt_embeds)
    raise RuntimeError("encode_prompt returned unexpected structure")


def get_text_context(model_dir: str, prompt: str, negative: Optional[str], *, device: str, dtype: str, text_encoder_dir: Optional[str] = None, tokenizer_dir: Optional[str] = None, vae_dir: Optional[str] = None, model_key: Optional[str] = None):
    try:
        # Resolve tokenizer automatically if not provided or invalid
        tk_dir = tokenizer_dir
        te_dir = text_encoder_dir
        try:
            from .tokenizer_resolver import ensure_tokenizer, ensure_text_encoder
            cand = [tokenizer_dir, text_encoder_dir, os.path.join(model_dir, 'tokenizer'), model_dir]
            tk_path, _info_tk = ensure_tokenizer(model_key=model_key, candidates=cand)
            if tk_path:
                tk_dir = tk_path
            # Resolve text encoder similarly (may download weights)
            ecand = [text_encoder_dir, os.path.join(model_dir, 'text_encoder'), model_dir]
            te_path, _info_te = ensure_text_encoder(model_key=model_key, candidates=ecand)
            if te_path:
                te_dir = te_path
        except Exception:
            pass
        return encode_with_text_encoder(model_dir, prompt, negative, device=device, dtype=dtype, text_encoder_dir=te_dir, tokenizer_dir=tk_dir)
    except Exception:
        return encode_with_pipeline(model_dir, prompt, negative, device=device, dtype=dtype, vae_dir=vae_dir)
