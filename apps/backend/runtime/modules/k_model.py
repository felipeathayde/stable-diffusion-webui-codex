import torch
import logging

from apps.backend.runtime import attention
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.modules.k_prediction import k_prediction_from_diffusers_scheduler


logger = logging.getLogger("backend.runtime.k_model")


class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        logger.info('K-Model Created: storage_dtype=%s computation_dtype=%s', self.storage_dtype, self.computation_dtype)

        self.diffusion_model = model

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        # Invariants: context and optional y must be consistent with diffusion model config
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError(f"UNet context must be a 3D tensor (B,S,C); got {type(context).__name__} shape={getattr(context,'shape',None)}")

        # Derive expected context dims from codex_config when available
        expected_ctx_dim = None
        cfg = getattr(self.diffusion_model, "codex_config", None)
        if cfg is not None:
            cd = getattr(cfg, "context_dim", None)
            if isinstance(cd, int):
                expected_ctx_dim = cd
            elif isinstance(cd, (list, tuple)) and len(cd) > 0:
                # If multiple values are present, require the feature dim to be one of them
                expected_ctx_dim = set(int(v) for v in cd if isinstance(v, int))

        feat_dim = int(context.shape[-1])
        if isinstance(expected_ctx_dim, int) and feat_dim != expected_ctx_dim:
            raise ValueError(
                f"UNet context feature dim mismatch: got {feat_dim}, expected {expected_ctx_dim}. "
                f"Hint: check SDXL concatenation (should be 2048) and UNet context_dim."
            )
        if isinstance(expected_ctx_dim, set) and expected_ctx_dim and feat_dim not in expected_ctx_dim:
            raise ValueError(
                f"UNet context feature dim {feat_dim} not in allowed set {sorted(expected_ctx_dim)}."
            )

        # If the UNet expects class/ADM conditioning, ensure 'y' is present in kwargs
        # Note: Flux uses pooled vector differently than SDXL ADM, so we only error if
        # the model needs y but doesn't have it. Extra y for flow models is OK.
        needs_y = getattr(self.diffusion_model, "num_classes", None) is not None
        has_y = "y" in kwargs and isinstance(kwargs["y"], torch.Tensor)
        if needs_y and not has_y:
            raise ValueError(
                f"UNet ADM conditioning mismatch: num_classes={getattr(self.diffusion_model,'num_classes',None)} "
                f"but y_present={has_y}. Ensure SDXL pooled vector is wired as 'y'."
            )

        # If present, enforce y feature size to match ADM channels declared in config
        if needs_y and has_y:
            y = kwargs["y"]
            adm_channels = None
            inner_cfg = getattr(self.diffusion_model, "codex_config", None)
            if inner_cfg is not None:
                adm_channels = getattr(inner_cfg, "adm_in_channels", None)
            if isinstance(adm_channels, int) and adm_channels > 0 and int(y.shape[1]) != adm_channels:
                raise ValueError(
                    f"UNet ADM feature mismatch: got y.shape[1]={int(y.shape[1])}, expected adm_in_channels={adm_channels}. "
                    f"Hint: SDXL vector should be [pooled_g, time_ids(6*256)], typically 1280+1536=2816."
                )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "apply_model: x=%s t=%s context=%s y=%s dtype=%s",
                tuple(x.shape), tuple(t.shape) if hasattr(t, 'shape') else (1,), tuple(context.shape),
                getattr(kwargs.get('y', None), 'shape', None), str(dtype)
            )

        model_output = self.diffusion_model(
            xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds
        ).float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        area = input_shape[0] * input_shape[2] * input_shape[3]
        dtype_size = memory_management.dtype_size(self.computation_dtype)

        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            scaler = 1.28
        else:
            scaler = 1.65
            if attention.get_attn_precision() == torch.float32:
                dtype_size = 4

        return scaler * area * dtype_size * 16384

    def forward(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """Standard forward method that delegates to apply_model.
        
        This is required by the memory management system which expects a 'forward' method.
        """
        return self.apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn, 
                                control=control, transformer_options=transformer_options, **kwargs)
