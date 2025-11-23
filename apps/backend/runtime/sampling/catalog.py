from __future__ import annotations

from typing import Dict, List, Set


SAMPLER_OPTIONS: List[Dict[str, object]] = [
    {"name": "automatic", "label": "Automatic", "aliases": ["auto", ""], "supported": True},
    {"name": "euler", "label": "Euler", "aliases": ["k_euler", "euler"], "supported": True},
    {"name": "euler a", "label": "Euler a", "aliases": ["k_euler_a", "euler a", "euler_ancestral"], "supported": True},
    {"name": "ddim", "label": "DDIM", "aliases": ["ddim"], "supported": True},
    {"name": "dpm++ 2m", "label": "DPM++ 2M", "aliases": ["dpmpp_2m", "dpm++ 2m"], "supported": True},
    {"name": "dpm++ 2m sde", "label": "DPM++ 2M SDE", "aliases": ["dpmpp_2m_sde", "dpm++ 2m sde"], "supported": True},
    {"name": "plms", "label": "PLMS", "aliases": ["lms"], "supported": True},
    {"name": "pndm", "label": "PNDM", "aliases": ["pndm"], "supported": True},
    {"name": "uni-pc", "label": "UniPC", "aliases": ["uni_pc", "unipc"], "supported": True},
    {"name": "heun", "label": "Heun", "aliases": ["heun"], "supported": False},
    {"name": "heunpp2", "label": "Heun++ 2", "aliases": ["heunpp2"], "supported": False},
    {"name": "dpm_2", "label": "DPM 2", "aliases": ["dpm_2"], "supported": False},
    {"name": "dpm_2_ancestral", "label": "DPM 2 Ancestral", "aliases": ["dpm_2_ancestral"], "supported": False},
    {"name": "dpm fast", "label": "DPM Fast", "aliases": ["dpm_fast"], "supported": False},
    {"name": "dpm adaptive", "label": "DPM Adaptive", "aliases": ["dpm_adaptive"], "supported": False},
    {"name": "dpm++ 2s ancestral", "label": "DPM++ 2S Ancestral", "aliases": ["dpmpp_2s_ancestral"], "supported": False},
    {"name": "dpm++ 2s ancestral cfgpp", "label": "DPM++ 2S Ancestral CFG++", "aliases": ["dpmpp_2s_ancestral_cfg_pp"], "supported": False},
    {"name": "dpm++ sde", "label": "DPM++ SDE", "aliases": ["dpmpp_sde", "dpmpp_sde_gpu"], "supported": False},
    {"name": "dpm++ 2m cfgpp", "label": "DPM++ 2M CFG++", "aliases": ["dpmpp_2m_cfg_pp"], "supported": False},
    {"name": "dpm++ 2m sde gpu", "label": "DPM++ 2M SDE GPU", "aliases": ["dpmpp_2m_sde_gpu"], "supported": False},
    {"name": "dpm++ 2m sde heun", "label": "DPM++ 2M SDE Heun", "aliases": ["dpmpp_2m_sde_heun"], "supported": False},
    {"name": "dpm++ 2m sde heun gpu", "label": "DPM++ 2M SDE Heun GPU", "aliases": ["dpmpp_2m_sde_heun_gpu"], "supported": False},
    {"name": "dpm++ 3m sde", "label": "DPM++ 3M SDE", "aliases": ["dpmpp_3m_sde", "dpmpp_3m_sde_gpu"], "supported": False},
    {"name": "ddpm", "label": "DDPM", "aliases": ["ddpm"], "supported": False},
    {"name": "lcm", "label": "LCM", "aliases": ["lcm"], "supported": False},
    {"name": "ipndm", "label": "iPNDM", "aliases": ["ipndm"], "supported": False},
    {"name": "ipndm_v", "label": "iPNDM v", "aliases": ["ipndm_v"], "supported": False},
    {"name": "deis", "label": "DEIS", "aliases": ["deis"], "supported": False},
    {"name": "res_multistep", "label": "Res MultiStep", "aliases": ["res_multistep", "res_multistep_cfg_pp"], "supported": False},
    {"name": "res_multistep_ancestral", "label": "Res MultiStep Ancestral", "aliases": ["res_multistep_ancestral", "res_multistep_ancestral_cfg_pp"], "supported": False},
    {"name": "gradient_estimation", "label": "Gradient Estimation", "aliases": ["gradient_estimation", "gradient_estimation_cfg_pp"], "supported": False},
    {"name": "er_sde", "label": "ER SDE", "aliases": ["er_sde"], "supported": False},
    {"name": "seeds_2", "label": "Seeds 2", "aliases": ["seeds_2"], "supported": False},
    {"name": "seeds_3", "label": "Seeds 3", "aliases": ["seeds_3"], "supported": False},
    {"name": "sa_solver", "label": "SA-Solver", "aliases": ["sa_solver"], "supported": False},
    {"name": "sa_solver_pece", "label": "SA-Solver PECE", "aliases": ["sa_solver_pece"], "supported": False},
    {"name": "uni_pc_bh2", "label": "UniPC (BH2)", "aliases": ["uni_pc_bh2"], "supported": False},
]

SAMPLER_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for entry in SAMPLER_OPTIONS:
    canonical = entry["name"]
    aliases = entry.get("aliases", []) if isinstance(entry.get("aliases"), list) else []
    for alias in [canonical, *aliases]:
        SAMPLER_ALIAS_TO_CANONICAL[alias.strip().lower()] = canonical  # type: ignore[arg-type]

SUPPORTED_SAMPLERS: Set[str] = {entry["name"] for entry in SAMPLER_OPTIONS if entry.get("supported", True)}

SCHEDULER_OPTIONS: List[Dict[str, object]] = [
    {
        "name": "automatic",
        "label": "Automatic",
        "aliases": ["auto", "", "use same scheduler", "use same", "same", "default"],
        "supported": True,
    },
    {
        "name": "karras",
        "label": "Karras",
        "aliases": ["karras"],
        "supported": True,
    },
    {
        "name": "exponential",
        "label": "Exponential",
        "aliases": ["exp", "exponential"],
        "supported": True,
    },
    {
        "name": "simple",
        "label": "Simple",
        "aliases": ["simple", "linear"],
        "supported": True,
    },
    {
        "name": "euler_discrete",
        "label": "Euler (discrete)",
        "aliases": [
            "euler",
            "euler a",
            "eulerdiscretescheduler",
            "eulerancestraldiscretescheduler",
        ],
        "supported": True,
    },
    {
        "name": "sgm_uniform",
        "label": "SGM Uniform",
        "aliases": ["sgm_uniform"],
        "supported": False,
    },
    {
        "name": "ddim_uniform",
        "label": "DDIM Uniform",
        "aliases": ["ddim_uniform"],
        "supported": False,
    },
    {
        "name": "beta",
        "label": "Beta",
        "aliases": ["beta"],
        "supported": False,
    },
    {
        "name": "normal",
        "label": "Normal",
        "aliases": ["normal"],
        "supported": False,
    },
    {
        "name": "linear_quadratic",
        "label": "Linear-Quadratic",
        "aliases": ["linear_quadratic"],
        "supported": False,
    },
    {
        "name": "kl_optimal",
        "label": "KL Optimal",
        "aliases": ["kl_optimal"],
        "supported": False,
    },
]

SCHEDULER_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for entry in SCHEDULER_OPTIONS:
    canonical = entry["name"]
    aliases = entry.get("aliases", []) if isinstance(entry.get("aliases"), list) else []
    for alias in [canonical, *aliases]:
        SCHEDULER_ALIAS_TO_CANONICAL[alias.strip().lower()] = canonical  # type: ignore[arg-type]

SUPPORTED_SCHEDULERS: Set[str] = {entry["name"] for entry in SCHEDULER_OPTIONS if entry.get("supported", True)}

# Default scheduler per sampler when user selects "Automatic" or "Use same".
SAMPLER_DEFAULT_SCHEDULER: Dict[str, str] = {
    "euler": "euler_discrete",
    "euler a": "euler_discrete",
    "dpm++ 2m": "karras",
    "dpm++ sde": "karras",
    "dpm++ 2m sde": "exponential",
    "dpm++ 2m sde heun": "exponential",
    "dpm++ 2s a": "karras",
    "dpm++ 3m sde": "exponential",
    "dpm2": "karras",
    "dpm2 a": "karras",
    "restart": "karras",
}

AUTO_TOKENS: Set[str] = {"automatic", "auto", "use same", "use same scheduler", "same", "default", ""}

__all__ = [
    "SAMPLER_OPTIONS",
    "SAMPLER_ALIAS_TO_CANONICAL",
    "SUPPORTED_SAMPLERS",
    "SCHEDULER_OPTIONS",
    "SCHEDULER_ALIAS_TO_CANONICAL",
    "SUPPORTED_SCHEDULERS",
    "SAMPLER_DEFAULT_SCHEDULER",
    "AUTO_TOKENS",
]
