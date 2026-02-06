---
license: other
license_name: circlestone-labs-non-commercial-license
license_link: LICENSE.md
tags:
  - diffusion-single-file
  - comfyui
---

<img src="example.png" width="400">

Anima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org. It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.

It is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.

This preview version is an intermediate model checkpoint. The model is still training and the final version will improve, especially for fine details and overall aesthetics.

# Installing and running
The model is natively supported in ComfyUI. The above image contains a workflow; you can open it in ComfyUI or drag-and-drop to get the workflow. The model files go in their respective folders inside your model directory:
- anima-preview.safetensors goes in ComfyUI/models/diffusion_models
- qwen_3_06b_base.safetensors goes in ComfyUI/models/text_encoders
- qwen_image_vae.safetensors goes in ComfyUI/models/vae (this is the Qwen-Image VAE, you might already have it)

## Generation settings
- The preview version should be used at about 1MP resolution. E.g. 1024x1024, 896x1152, 1152x896, etc.
- 30-50 steps, CFG 4-5.
- A variety of samplers work. Some of my favorites:
  - er_sde: neutral style, flat colors, sharp lines. I use this as a reasonable default.
  - euler_a: Softer, thinner lines. Can sometimes tend towards a 2.5D look. CFG can be pushed a bit higher than other samplers without burning the image.
  - dpmpp_2m_sde_gpu: similar in style to er_sde but can produce more variety and be more "creative". Depending on the prompt it can get too wild sometimes.

# Prompting
The model is trained on Danbooru-style tags, natural language captions, and combinations of tags and captions.

## Tag order
[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] [series] [artist] [general tags]

Within each tag section, the tags can be in arbitrary order.

## Quality tags
Human score based: masterpiece, best quality, good quality, normal quality, low quality, worst quality

PonyV7 aesthetic model based: score_9, score_8, ..., score_1

You can use either the human score quality tags, the aesthetic model tags, both together, or neither. All combinations work.

## Time period tags
Specific year: year 2025, year 2024, ...

Period: newest, recent, mid, early, old

## Meta tags
highres, absurdres, anime screenshot, jpeg artifacts, official art, etc

## Safety tags
safe, sensitive, nsfw, explicit

## Artist tags
Prefix artist with @. E.g. "@big chungus". **You must put @ in front of the artist.** The effect will be very weak if you don't.

## Full tag example
year 2025, newest, normal quality, score_5, highres, safe, 1girl, oomuro sakurako, yuru yuri, @nnn yryr, smile, brown hair, hat, solo, fur-trimmed gloves, open mouth, long hair, gift box, fang, skirt, red gloves, blunt bangs, gloves, one eye closed, shirt, brown eyes, santa costume, red hat, skin fang, twitter username, white background, holding bag, fur trim, simple background, brown skirt, bag, gift bag, looking at viewer, santa hat, ;d, red shirt, box, gift, fur-trimmed headwear, holding, red capelet, holding box, capelet

## Tag dropout
The model was trained with random tag dropout. You don't need to include every single relevant tag for the image.

## Dataset tags
To improve style and content diversity, the model was additionally trained on two non-anime datasets: LAION-POP (specifically the ye-pop version) and DeviantArt. Both were filtered to exclude photos. Because these datasets are qualitatively different from anime datasets, captions from them have been labeled with a "dataset tag". This occurs at the very beginning of a prompt followed by a newline. Optionally, the second line can contain either the image alt-text (ye-pop) or the title of the work (DeviantArt). Examples:


ye-pop<br>
For Sale: Others by Arun Prem<br>
Abstract, oil painting of three faceless, blue-skinned figures. Left: white, draped figure; center: yellow-shirted, dark-haired figure; right: red-veiled, dark-haired figure carrying another. Bold, textured colors, minimalist style.


deviantart<br>
Flame<br>
Digital painting of a fiery dragon with glowing yellow eyes, black horns, and a long, sinuous tail, perched on a glowing, molten rock formation. The background is a gradient of dark purple to orange.

## Natural language prompting tips
- If using pure natural langauge, more descriptive is better. Aim for at least 2 sentences. Extremely short prompts can give unexpected results (this will be better in the final version).
- You can mix tags and natural language in arbitrary order.
- You can put quality / artist tags at the beginning of a natural language prompt.
  - "masterpiece, best quality, @big chungus. An anime girl with medium-length blonde hair is..."
- Name a character, then describe their basic appearance.
  - "Digital artwork of Fern from Sousou no Frieren, with long purple hair and purple eyes, wearing a black coat over a white dress with puffy sleeves..."
  - This is extra important when prompting for multiple characters. If you just list off character names with no description of appearance, the model can get confused.

# Model comparison
You may be interested in comparing Anima's outputs with other models. A ComfyUI workflow, anima_comparison.json, is provided. This workflow generates a grid of images where each model is a column and the rows are different seeds. It can be configured to compare any number of models you select by changing a few output nodes. Supported model architectures: Anima, SDXL, Lumina, Chroma, Newbie-Image. The default configuration compares Anima, NetaYume, and Newbie-Image.

# Limitations
- The model doesn't do realism well. This is intended. It is an anime / illustration / art focused model.
- The model may generate undesired content, especially if the prompt is short or lacking details.
  - Avoid this by using the appropriate safety tags in the positive and negative prompts, and by writing sufficiently detailed prompts.
- The model isn't great at text rendering. It can generally do single words and sometimes short phrases, but lengthy text rendering won't work well.
- The preview model isn't that good at higher resolutions yet.
  - It is a medium-resolution intermediate checkpoint, trained on a small amount of high-res images.
  - The final version will have been trained on a dedicated high-res phase. Details and overall image composition will improve.
- The preview model is a true base model. It hasn't been aesthetic tuned on a curated dataset. The default style is very plain and neutral, which is especially apparent if you don't use artist or quality tags.

# License
This model is licensed under the CircleStone Labs Non-Commercial License. The model and derivatives are only usable for non-commercial purposes. Additionally, this model constitutes a "Derivative Model" of Cosmos-Predict2-2B-Text2Image, and therefore is subject to the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) insofar as it applies to Derivative Models.

The details of the commercial licensing process are still being worked out. For now, you can express your interest in acquiring a commercial license by emailing tdrussell1@proton.me

Built on NVIDIA Cosmos.