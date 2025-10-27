"""
Vendored copy of huggingface_guess.

Upstream: https://github.com/lllyasviel/huggingface_guess
Commit: 70942022b6bcd17d941c1b4172804175758618e2 (2024-10-29)
Note: Original project is licensed under GPL-3.0. See VENDORED.md for details.
"""

from huggingface_guess.detection import model_config_from_unet, unet_prefix_from_state_dict, model_config_from_diffusers_unet


def guess(state_dict):
    unet_key_prefix = unet_prefix_from_state_dict(state_dict)
    result = model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False)
    result.unet_key_prefix = [unet_key_prefix]
    if 'image_model' in result.unet_config:
        del result.unet_config['image_model']
    if 'audio_model' in result.unet_config:
        del result.unet_config['audio_model']
    return result


def guess_repo_name(state_dict):
    config = guess(state_dict)
    assert config is not None
    repo_id = config.huggingface_repo
    return repo_id

