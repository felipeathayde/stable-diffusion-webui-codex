# Auto-generated from settings_schema.json. Do not edit by hand.
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import Any, Optional

class SettingType(StrEnum):
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    NUMBER = "number"
    TEXT = "text"
    COLOR = "color"
    HTML = "html"

@dataclass
class CategoryDef:
    id: "CategoryId"
    label: str

@dataclass
class SectionDef:
    key: "SectionId"
    label: str
    category_id: Optional["CategoryId"]

@dataclass
class FieldDef:
    key: str
    label: str
    type: SettingType
    section: "SectionId"
    default: Any | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    choices_source: str | None = None

def schema_to_json() -> dict:
    return {
        "categories": [asdict(c) | {"id": c.id.value} for c in CATEGORIES],
        "sections": [asdict(s) | {"key": s.key.value, "category_id": s.category_id.value if s.category_id else None} for s in SECTIONS],
        "fields": [
            {k: (v.value if hasattr(v, "value") else v) for k, v in (asdict(f) | {"type": f.type.value, "section": f.section.value}).items()}
            for f in FIELDS
        ],
        "version": 1,
        "source": "settings_registry.py"
    }

def field_index() -> dict[str, FieldDef]:
    return {f.key: f for f in FIELDS}



class CategoryId(StrEnum):
    SAVING = 'saving'
    SD = 'sd'
    UI = 'ui'
    SYSTEM = 'system'
    POSTPROCESSING = 'postprocessing'
    TRAINING = 'training'

CATEGORIES: list[CategoryDef] = [
    CategoryDef(CategoryId.SAVING, 'Saving images'),
    CategoryDef(CategoryId.SD, 'Stable Diffusion'),
    CategoryDef(CategoryId.UI, 'User Interface'),
    CategoryDef(CategoryId.SYSTEM, 'System'),
    CategoryDef(CategoryId.POSTPROCESSING, 'Postprocessing'),
    CategoryDef(CategoryId.TRAINING, 'Training'),
]


class SectionId(StrEnum):
    SAVING_IMAGES = 'saving-images'
    SAVING_PATHS = 'saving-paths'
    SAVING_TO_DIRS = 'saving-to-dirs'
    UPSCALING = 'upscaling'
    FACE_RESTORATION = 'face-restoration'
    SYSTEM = 'system'
    PROFILER = 'profiler'
    API = 'API'
    TRAINING = 'training'
    SD = 'sd'
    SDXL = 'sdxl'
    SD3 = 'sd3'
    VAE = 'vae'
    IMG2IMG = 'img2img'
    OPTIMIZATIONS = 'optimizations'
    COMPATIBILITY = 'compatibility'
    INTERROGATE = 'interrogate'
    EXTRA_NETWORKS = 'extra_networks'
    UI_PROMPT_EDITING = 'ui_prompt_editing'
    UI_GALLERY = 'ui_gallery'
    UI_ALTERNATIVES = 'ui_alternatives'
    UI = 'ui'
    INFOTEXT = 'infotext'
    SAMPLER_PARAMS = 'sampler-params'
    POSTPROCESSING = 'postprocessing'

SECTIONS: list[SectionDef] = [
    SectionDef(SectionId.SAVING_IMAGES, 'Saving images/grids', CategoryId.SAVING),
    SectionDef(SectionId.SAVING_PATHS, 'Paths for saving', CategoryId.SAVING),
    SectionDef(SectionId.SAVING_TO_DIRS, 'Saving to a directory', CategoryId.SAVING),
    SectionDef(SectionId.UPSCALING, 'Upscaling', CategoryId.POSTPROCESSING),
    SectionDef(SectionId.FACE_RESTORATION, 'Face restoration', CategoryId.POSTPROCESSING),
    SectionDef(SectionId.SYSTEM, 'System', CategoryId.SYSTEM),
    SectionDef(SectionId.PROFILER, 'Profiler', CategoryId.SYSTEM),
    SectionDef(SectionId.API, 'API', CategoryId.SYSTEM),
    SectionDef(SectionId.TRAINING, 'Training', CategoryId.TRAINING),
    SectionDef(SectionId.SD, 'Stable Diffusion', CategoryId.SD),
    SectionDef(SectionId.SDXL, 'Stable Diffusion XL', CategoryId.SD),
    SectionDef(SectionId.SD3, 'Stable Diffusion 3', CategoryId.SD),
    SectionDef(SectionId.VAE, 'VAE', CategoryId.SD),
    SectionDef(SectionId.IMG2IMG, 'img2img', CategoryId.SD),
    SectionDef(SectionId.OPTIMIZATIONS, 'Optimizations', CategoryId.SD),
    SectionDef(SectionId.COMPATIBILITY, 'Compatibility', CategoryId.SD),
    SectionDef(SectionId.INTERROGATE, 'Interrogate', None),
    SectionDef(SectionId.EXTRA_NETWORKS, 'Extra Networks', CategoryId.SD),
    SectionDef(SectionId.UI_PROMPT_EDITING, 'Prompt editing', CategoryId.UI),
    SectionDef(SectionId.UI_GALLERY, 'Gallery', CategoryId.UI),
    SectionDef(SectionId.UI_ALTERNATIVES, 'UI alternatives', CategoryId.UI),
    SectionDef(SectionId.UI, 'User interface', CategoryId.UI),
    SectionDef(SectionId.INFOTEXT, 'Infotext', CategoryId.UI),
    SectionDef(SectionId.SAMPLER_PARAMS, 'Sampler parameters', CategoryId.SD),
    SectionDef(SectionId.POSTPROCESSING, 'Postprocessing', CategoryId.POSTPROCESSING),
]


FIELDS: list[FieldDef] = [
    FieldDef(key='samples_save', label='Always save all generated images', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='samples_format', label='File format for images', type=SettingType.TEXT, section=SectionId.SAVING_IMAGES, default='png', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_images_add_number', label='Add number to filename when saving', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_images_replace_action', label='Saving the image to an existing file', type=SettingType.RADIO, section=SectionId.SAVING_IMAGES, default='Replace', min=None, max=None, step=None, choices=['Replace', 'Add number suffix'], choices_source=None),
    FieldDef(key='grid_save', label='Always save all generated image grids', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_format', label='File format for grids', type=SettingType.TEXT, section=SectionId.SAVING_IMAGES, default='png', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_extended_filename', label='Add extended info (seed, prompt) to filename when saving grid', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_only_if_multiple', label='Do not save grids consisting of one picture', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_prevent_empty_spots', label='Prevent empty spots in grid (when set to autodetect)', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='n_rows', label='Grid row count; use -1 for autodetect and 0 for it to be same as batch size', type=SettingType.SLIDER, section=SectionId.SAVING_IMAGES, default=-1, min=-1, max=16, step=1, choices=None, choices_source=None),
    FieldDef(key='font', label='Font for image grids that have text', type=SettingType.TEXT, section=SectionId.SAVING_IMAGES, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_text_active_color', label='Text color for image grids', type=SettingType.COLOR, section=SectionId.SAVING_IMAGES, default='#000000', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_text_inactive_color', label='Inactive text color for image grids', type=SettingType.COLOR, section=SectionId.SAVING_IMAGES, default='#999999', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_background_color', label='Background color for image grids', type=SettingType.COLOR, section=SectionId.SAVING_IMAGES, default='#ffffff', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_images_before_face_restoration', label='Save a copy of image before doing face restoration.', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_images_before_highres_fix', label='Save a copy of image before applying highres fix.', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_images_before_color_correction', label='Save a copy of image before applying color correction to img2img results', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_mask', label='For inpainting, save a copy of the greyscale mask', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_mask_composite', label='For inpainting, save a masked composite', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='jpeg_quality', label='Quality for saved jpeg and avif images', type=SettingType.SLIDER, section=SectionId.SAVING_IMAGES, default=80, min=1, max=100, step=1, choices=None, choices_source=None),
    FieldDef(key='webp_lossless', label='Use lossless compression for webp images', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='img_downscale_threshold', label='File size limit for the above option, MB', type=SettingType.NUMBER, section=SectionId.SAVING_IMAGES, default=4.0, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='target_side_length', label='Width/height limit for the above option, in pixels', type=SettingType.NUMBER, section=SectionId.SAVING_IMAGES, default=4000, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_original_name_batch', label='Use original name for output filename during batch process in extras tab', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_upscaler_name_as_suffix', label='Use upscaler name as filename suffix in the extras tab', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_selected_only', label="When using \\'Save\\' button, only save a single selected image", type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_write_log_csv', label="Write log.csv when saving images using \\'Save\\' button", type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_init_img', label='Save init images when using img2img', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='temp_dir', label='Directory for temporary images; leave empty for default', type=SettingType.TEXT, section=SectionId.SAVING_IMAGES, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='clean_temp_dir_at_start', label='Cleanup non-default temporary directory when starting webui', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_samples', label='Output directory for images; if empty, defaults to three directories below', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_txt2img_samples', label='Output directory for txt2img images', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_img2img_samples', label='Output directory for img2img images', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_extras_samples', label='Output directory for images from extras tab', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_grids', label='Output directory for grids; if empty, defaults to two directories below', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_txt2img_grids', label='Output directory for txt2img grids', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_img2img_grids', label='Output directory for img2img grids', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_save', label='Directory for saving images using the Save button', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='outdir_init_images', label='Directory for saving init images when using img2img', type=SettingType.TEXT, section=SectionId.SAVING_PATHS, default='util.truncate_path', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_to_dirs', label='Save images to a subdirectory', type=SettingType.CHECKBOX, section=SectionId.SAVING_TO_DIRS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='grid_save_to_dirs', label='Save grids to a subdirectory', type=SettingType.CHECKBOX, section=SectionId.SAVING_TO_DIRS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_save_to_dirs_for_ui', label='When using "Save" button, save images to a subdirectory', type=SettingType.CHECKBOX, section=SectionId.SAVING_TO_DIRS, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='directories_max_prompt_words', label='Max prompt words for [prompt_words] pattern', type=SettingType.SLIDER, section=SectionId.SAVING_TO_DIRS, default=8, min=1, max=20, step=1, choices=None, choices_source=None),
    FieldDef(key='realesrgan_enabled_models', label='Select which Real-ESRGAN models to show in the web UI.', type=SettingType.TEXT, section=SectionId.UPSCALING, default=['R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B'], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='dat_enabled_models', label='Select which DAT models to show in the web UI.', type=SettingType.TEXT, section=SectionId.UPSCALING, default=['DAT x2', 'DAT x3', 'DAT x4'], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='upscaler_for_img2img', label='Upscaler for img2img', type=SettingType.DROPDOWN, section=SectionId.UPSCALING, default=None, min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='set_scale_by_when_changing_upscaler', label='Automatically set the Scale by factor based on the name of the selected Upscaler.', type=SettingType.CHECKBOX, section=SectionId.UPSCALING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='face_restoration_model', label='Face restoration model', type=SettingType.RADIO, section=SectionId.FACE_RESTORATION, default='CodeFormer', min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='face_restoration_unload', label='Move face restoration model from VRAM into RAM after processing', type=SettingType.CHECKBOX, section=SectionId.FACE_RESTORATION, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='auto_launch_browser', label='Automatically open webui in browser on startup', type=SettingType.RADIO, section=SectionId.SYSTEM, default='Local', min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='enable_console_prompts', label='Print prompts to console when generating with txt2img and img2img.', type=SettingType.TEXT, section=SectionId.SYSTEM, default='shared.cmd_opts.enable_console_prompts', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='samples_log_stdout', label='Always print all generation info to standard output', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='multiple_tqdm', label='Add a second progress bar to the console that shows progress for an entire job.', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='enable_upscale_progressbar', label='Show a progress bar in the console for tiled upscaling.', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='print_hypernet_extra', label='Print extra hypernetwork information to console.', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='hide_ldm_prints', label="Prevent Stability-AI\\'s ldm/sgm modules from printing noise to console.", type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='dump_stacks_on_signal', label='Print stack traces before exiting the program with ctrl+c.', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_explanation', label='', type=SettingType.TEXT, section=SectionId.PROFILER, default='\nThose settings allow you to enable torch profiler when generating pictures.\nProfiling allows you to see which code uses how much of computer\'s resources during generation.\nEach generation writes its own profile to one file, overwriting previous.\nThe file can be viewed in <a href="chrome:tracing">Chrome</a>, or on a <a href="https://ui.perfetto.dev/">Perfetto</a> web site.\nWarning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB in size.\n', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_enable', label='Enable profiling', type=SettingType.CHECKBOX, section=SectionId.PROFILER, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_activities', label='Activities', type=SettingType.TEXT, section=SectionId.PROFILER, default=['CPU'], min=None, max=None, step=None, choices=['CPU', 'CUDA'], choices_source=None),
    FieldDef(key='profiling_record_shapes', label='Record shapes', type=SettingType.CHECKBOX, section=SectionId.PROFILER, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_profile_memory', label='Profile memory', type=SettingType.CHECKBOX, section=SectionId.PROFILER, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_with_stack', label='Include python stack', type=SettingType.CHECKBOX, section=SectionId.PROFILER, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='profiling_filename', label='Profile filename', type=SettingType.TEXT, section=SectionId.PROFILER, default='trace.json', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='api_enable_requests', label='Allow http:// and https:// URLs for input images in API', type=SettingType.CHECKBOX, section=SectionId.API, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='api_forbid_local_requests', label='Forbid URLs to local resources', type=SettingType.CHECKBOX, section=SectionId.API, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='api_useragent', label='User agent for requests', type=SettingType.TEXT, section=SectionId.API, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='unload_models_when_training', label='Move VAE and CLIP to RAM when training if possible. Saves VRAM.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='pin_memory', label='Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_optimizer_state', label='Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_training_settings_to_txt', label='Save textual inversion and hypernet settings to a text file whenever training starts.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='dataset_filename_word_regex', label='Filename word regex', type=SettingType.TEXT, section=SectionId.TRAINING, default='', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='dataset_filename_join_string', label='Filename join string', type=SettingType.TEXT, section=SectionId.TRAINING, default=' ', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_image_repeats_per_epoch', label='Number of repeats for a single input image per epoch; used only for displaying epoch number', type=SettingType.NUMBER, section=SectionId.TRAINING, default=1, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_write_csv_every', label='Save an csv containing the loss to log directory every N steps, 0 to disable', type=SettingType.NUMBER, section=SectionId.TRAINING, default=500, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_xattention_optimizations', label='Use cross attention optimizations while training', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_enable_tensorboard', label='Enable tensorboard logging.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_tensorboard_save_images', label='Save generated images within tensorboard.', type=SettingType.CHECKBOX, section=SectionId.TRAINING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='training_tensorboard_flush_every', label='How often, in seconds, to flush the pending tensorboard events and summaries to disk.', type=SettingType.NUMBER, section=SectionId.TRAINING, default=120, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='sd_model_checkpoint', label='(Managed by Codex)', type=SettingType.TEXT, section=SectionId.SD, default=None, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='sd_checkpoints_limit', label='Maximum number of checkpoints loaded at the same time', type=SettingType.SLIDER, section=SectionId.SD, default=1, min=1, max=10, step=1, choices=None, choices_source=None),
    FieldDef(key='enable_batch_seeds', label='Make K-diffusion samplers produce same images in a batch as when making a single image', type=SettingType.CHECKBOX, section=SectionId.SD, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='CLIP_stop_at_last_layers', label='(Managed by Codex)', type=SettingType.NUMBER, section=SectionId.SD, default=1, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='upcast_attn', label='Upcast cross attention layer to float32', type=SettingType.CHECKBOX, section=SectionId.SD, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='hires_fix_refiner_pass', label='Hires fix: which pass to enable refiner for', type=SettingType.RADIO, section=SectionId.SD, default='second pass', min=None, max=None, step=None, choices=['first pass', 'second pass', 'both passes'], choices_source=None),
    FieldDef(key='sdxl_crop_top', label='crop top coordinate', type=SettingType.NUMBER, section=SectionId.SDXL, default=0, min=0, max=1024, step=1, choices=None, choices_source=None),
    FieldDef(key='sdxl_crop_left', label='crop left coordinate', type=SettingType.NUMBER, section=SectionId.SDXL, default=0, min=0, max=1024, step=1, choices=None, choices_source=None),
    FieldDef(key='sd_vae_explanation', label='', type=SettingType.TEXT, section=SectionId.VAE, default="\n<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.\n", min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='sd_vae_checkpoint_cache', label='VAE Checkpoints to cache in RAM', type=SettingType.SLIDER, section=SectionId.VAE, default=0, min=0, max=10, step=1, choices=None, choices_source=None),
    FieldDef(key='sd_vae', label='(Managed by Codex)', type=SettingType.TEXT, section=SectionId.VAE, default='Automatic', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='inpainting_mask_weight', label='Inpainting conditioning mask strength', type=SettingType.SLIDER, section=SectionId.IMG2IMG, default=1.0, min=0.0, max=1.0, step=0.01, choices=None, choices_source=None),
    FieldDef(key='initial_noise_multiplier', label='Noise multiplier for img2img', type=SettingType.SLIDER, section=SectionId.IMG2IMG, default=1.0, min=0.0, max=1.5, step=0.001, choices=None, choices_source=None),
    FieldDef(key='img2img_color_correction', label='Apply color correction to img2img results to match original colors.', type=SettingType.CHECKBOX, section=SectionId.IMG2IMG, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='img2img_background_color', label='With img2img, fill transparent parts of the input image with this color.', type=SettingType.COLOR, section=SectionId.IMG2IMG, default='#ffffff', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='return_mask', label='For inpainting, include the greyscale mask in results for web', type=SettingType.CHECKBOX, section=SectionId.IMG2IMG, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='return_mask_composite', label='For inpainting, include masked composite in results for web', type=SettingType.CHECKBOX, section=SectionId.IMG2IMG, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='img2img_autosize', label='After loading into Img2img, automatically update Width and Height', type=SettingType.CHECKBOX, section=SectionId.IMG2IMG, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='cross_attention_optimization', label='Cross attention optimization', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='Automatic', min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='codex_core_device', label='Core device (UNet/primary)', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','cuda','cpu','mps','xpu','directml'], choices_source=None),
    FieldDef(key='codex_te_device', label='Text encoder device', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','cuda','cpu','mps','xpu','directml'], choices_source=None),
    FieldDef(key='codex_vae_device', label='VAE device', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','cuda','cpu','mps','xpu','directml'], choices_source=None),
    FieldDef(key='codex_core_dtype', label='Core dtype', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','fp16','bf16','fp32'], choices_source=None),
    FieldDef(key='codex_te_dtype', label='Text encoder dtype', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','fp16','bf16','fp32'], choices_source=None),
    FieldDef(key='codex_vae_dtype', label='VAE dtype', type=SettingType.DROPDOWN, section=SectionId.OPTIMIZATIONS, default='auto', min=None, max=None, step=None, choices=['auto','fp16','bf16','fp32'], choices_source=None),
    FieldDef(key='codex_smart_offload', label='Smart offload (TE/UNet/VAE)', type=SettingType.CHECKBOX, section=SectionId.OPTIMIZATIONS, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_smart_fallback', label='Smart fallback to CPU on OOM', type=SettingType.CHECKBOX, section=SectionId.OPTIMIZATIONS, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_smart_cache', label='Smart cache (TE/embeds)', type=SettingType.CHECKBOX, section=SectionId.OPTIMIZATIONS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_try_reproduce', label='Try to reproduce the results from external software', type=SettingType.RADIO, section=SectionId.COMPATIBILITY, default='None', min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='use_old_emphasis_implementation', label='Use old emphasis implementation. Can be useful to reproduce old seeds.', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_old_karras_scheduler_sigmas', label='Use old karras scheduler sigmas (0.1 to 10).', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='no_dpmpp_sde_batch_determinism', label='Do not make DPM++ SDE deterministic across different batch sizes.', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_old_hires_fix_width_height', label='For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to).', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='hires_fix_use_firstpass_conds', label='For hires fix, calculate conds of second pass using extra networks of first pass.', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='use_downcasted_alpha_bar', label='Downcast model alphas_cumprod to fp16 before sampling. For reproducing old seeds.', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='refiner_switch_by_sample_steps', label='Switch to refiner by sampling steps instead of model timesteps. Old behavior for refiner.', type=SettingType.CHECKBOX, section=SectionId.COMPATIBILITY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='interrogate_keep_models_in_memory', label='Keep models in VRAM', type=SettingType.CHECKBOX, section=SectionId.INTERROGATE, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='interrogate_clip_num_beams', label='BLIP: num_beams', type=SettingType.SLIDER, section=SectionId.INTERROGATE, default=1, min=1, max=16, step=1, choices=None, choices_source=None),
    FieldDef(key='interrogate_clip_min_length', label='BLIP: minimum description length', type=SettingType.SLIDER, section=SectionId.INTERROGATE, default=24, min=1, max=128, step=1, choices=None, choices_source=None),
    FieldDef(key='interrogate_clip_max_length', label='BLIP: maximum description length', type=SettingType.SLIDER, section=SectionId.INTERROGATE, default=48, min=1, max=256, step=1, choices=None, choices_source=None),
    FieldDef(key='interrogate_clip_skip_categories', label='CLIP: skip inquire categories', type=SettingType.TEXT, section=SectionId.INTERROGATE, default=[], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='interrogate_deepbooru_score_threshold', label='deepbooru: score threshold', type=SettingType.SLIDER, section=SectionId.INTERROGATE, default=0.5, min=0, max=1, step=0.01, choices=None, choices_source=None),
    FieldDef(key='extra_networks_default_multiplier', label='Default multiplier for extra networks', type=SettingType.SLIDER, section=SectionId.EXTRA_NETWORKS, default=1.0, min=0.0, max=2.0, step=0.01, choices=None, choices_source=None),
    FieldDef(key='extra_networks_card_show_desc', label='Show description on card', type=SettingType.CHECKBOX, section=SectionId.EXTRA_NETWORKS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='extra_networks_card_description_is_html', label='Treat card description as HTML', type=SettingType.CHECKBOX, section=SectionId.EXTRA_NETWORKS, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='textual_inversion_print_at_load', label='Print a list of Textual Inversion embeddings when loading model', type=SettingType.CHECKBOX, section=SectionId.EXTRA_NETWORKS, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='textual_inversion_add_hashes_to_infotext', label='Add Textual Inversion hashes to infotext', type=SettingType.CHECKBOX, section=SectionId.EXTRA_NETWORKS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='sd_hypernetwork', label='Add hypernetwork to prompt', type=SettingType.DROPDOWN, section=SectionId.EXTRA_NETWORKS, default='None', min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='keyedit_precision_attention', label='Precision for (attention:1.1) when editing the prompt with Ctrl+up/down', type=SettingType.SLIDER, section=SectionId.UI_PROMPT_EDITING, default=0.1, min=0.01, max=0.2, step=0.001, choices=None, choices_source=None),
    FieldDef(key='keyedit_precision_extra', label='Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down', type=SettingType.SLIDER, section=SectionId.UI_PROMPT_EDITING, default=0.05, min=0.01, max=0.2, step=0.001, choices=None, choices_source=None),
    FieldDef(key='keyedit_delimiters', label='Word delimiters when editing the prompt with Ctrl+up/down', type=SettingType.TEXT, section=SectionId.UI_PROMPT_EDITING, default='.,\\/!?%^*;:{}=`~() ', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='keyedit_delimiters_whitespace', label='Ctrl+up/down whitespace delimiters', type=SettingType.TEXT, section=SectionId.UI_PROMPT_EDITING, default=['Tab', 'Carriage Return', 'Line Feed'], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='keyedit_move', label='Alt+left/right moves prompt elements', type=SettingType.CHECKBOX, section=SectionId.UI_PROMPT_EDITING, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='disable_token_counters', label='Disable prompt token counters', type=SettingType.CHECKBOX, section=SectionId.UI_PROMPT_EDITING, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='return_grid', label='Show grid in gallery', type=SettingType.CHECKBOX, section=SectionId.UI_GALLERY, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='do_not_show_images', label='Do not show any images in gallery', type=SettingType.CHECKBOX, section=SectionId.UI_GALLERY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='js_modal_lightbox', label='Full page image viewer: enable', type=SettingType.CHECKBOX, section=SectionId.UI_GALLERY, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='js_modal_lightbox_initially_zoomed', label='Full page image viewer: show images zoomed in by default', type=SettingType.CHECKBOX, section=SectionId.UI_GALLERY, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='js_modal_lightbox_gamepad', label='Full page image viewer: navigate with gamepad', type=SettingType.CHECKBOX, section=SectionId.UI_GALLERY, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='open_dir_button_choice', label='What directory the [📂] button opens', type=SettingType.RADIO, section=SectionId.UI_GALLERY, default='Subdirectory', min=None, max=None, step=None, choices=['Output Root', 'Subdirectory', 'Subdirectory (even temp dir)'], choices_source=None),
    FieldDef(key='tabs_without_quick_settings_bar', label='UI tabs without Quicksettings bar (top row)', type=SettingType.TEXT, section=SectionId.UI, default=['Spaces'], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='show_progress_in_title', label='Show generation progress in window title.', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='send_seed', label='Send seed when sending prompt or image to other interface', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='send_size', label='Send size when sending prompt or image to another interface', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='infotext_explanation', label='', type=SettingType.TEXT, section=SectionId.INFOTEXT, default='\nInfotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.\nIt is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.\n', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='enable_pnginfo', label='Write infotext to metadata of the generated image', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='save_txt', label='Create a text file with infotext next to every generated image', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_model_name_to_info', label='Add model name to infotext', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_model_hash_to_info', label='Add model hash to infotext', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_vae_name_to_info', label='Add VAE name to infotext', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_vae_hash_to_info', label='Add VAE hash to infotext', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_user_name_to_info', label='Add user name to infotext when authenticated', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='add_version_to_infotext', label='Add program version to infotext', type=SettingType.CHECKBOX, section=SectionId.INFOTEXT, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='infotext_skip_pasting', label='Disregard fields from pasted infotext', type=SettingType.TEXT, section=SectionId.INFOTEXT, default=[], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='show_progressbar', label='Show progressbar', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='live_previews_enable', label='Show live previews of the created image', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='live_previews_image_format', label='Live preview file format', type=SettingType.RADIO, section=SectionId.UI, default='png', min=None, max=None, step=None, choices=['jpeg', 'png', 'webp'], choices_source=None),
    FieldDef(key='show_progress_grid', label='Show previews of all images generated in a batch as a grid', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='live_preview_content', label='Live preview subject', type=SettingType.RADIO, section=SectionId.UI, default='Prompt', min=None, max=None, step=None, choices=['Combined', 'Prompt', 'Negative prompt'], choices_source=None),
    FieldDef(key='js_live_preview_in_modal_lightbox', label='Show Live preview in full page image viewer', type=SettingType.CHECKBOX, section=SectionId.UI, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='prevent_screen_sleep_during_generation', label='Prevent screen sleep during generation', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='ddim_discretize', label='img2img DDIM discretize', type=SettingType.RADIO, section=SectionId.SAMPLER_PARAMS, default='uniform', min=None, max=None, step=None, choices=['uniform', 'quad'], choices_source=None),
    FieldDef(key='uni_pc_variant', label='UniPC variant', type=SettingType.RADIO, section=SectionId.SAMPLER_PARAMS, default='bh1', min=None, max=None, step=None, choices=['bh1', 'bh2', 'vary_coeff'], choices_source=None),
    FieldDef(key='uni_pc_skip_type', label='UniPC skip type', type=SettingType.RADIO, section=SectionId.SAMPLER_PARAMS, default='time_uniform', min=None, max=None, step=None, choices=['time_uniform', 'time_quadratic', 'logSNR'], choices_source=None),
    FieldDef(key='uni_pc_lower_order_final', label='UniPC lower order final', type=SettingType.CHECKBOX, section=SectionId.SAMPLER_PARAMS, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='postprocessing_enable_in_main_ui', label='Enable postprocessing operations in txt2img and img2img tabs', type=SettingType.TEXT, section=SectionId.POSTPROCESSING, default=[], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='postprocessing_disable_in_extras', label='Disable postprocessing operations in extras tab', type=SettingType.TEXT, section=SectionId.POSTPROCESSING, default=[], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='postprocessing_operation_order', label='Postprocessing operation order', type=SettingType.TEXT, section=SectionId.POSTPROCESSING, default=[], min=None, max=None, step=None, choices=None, choices_source='lambda'),
    FieldDef(key='upscaling_max_images_in_cache', label='Maximum number of images in upscaling cache', type=SettingType.SLIDER, section=SectionId.POSTPROCESSING, default=5, min=0, max=10, step=1, choices=None, choices_source=None),
]
