# WAN 2.2 Video — Quickstart

WAN 2.2 lives in dedicated model tabs and uses the video pipeline (txt2vid/img2vid) instead of the SDXL image workspace.

## Prerequisites

- Place your WAN 2.2 `.gguf` files under a directory such as:
  - `models/wan22/High/…`
  - `models/wan22/Low/…`
- Make sure the repo is running with WAN engines enabled (check the Models tab for WAN 2.2 inventory).

## Creating a WAN tab

You can enter WAN 2.2 in two ways; both end up on `/models/:tabId`:

1. From **Home** (`/`):
   - In the “Create Model Tab” panel, choose engine type **WAN 2.2**.
   - Give the tab an optional title (for example, “WAN 2.2 — default”).
   - Click **Create** to jump straight into the new tab.
2. From **Models**:
   - Open the **Models** tab from the top navigation.
   - Use the “New tab” controls and select the **WAN** engine type.

You can create multiple WAN tabs, for example:

- One per model directory (different GGUF sets).
- One per output profile (resolution/FPS/bitrate).

## Configuring WAN models

Inside the WAN tab:

- Fill **High stage model dir** (typically pointing to `*High*.gguf` files).
- Fill **Low stage model dir** (typically pointing to `*Low*.gguf` files).
- Leave the placeholders as a guide (`/models/wan22/*High*.gguf`, `/models/wan22/*Low*.gguf`) and adapt them to your layout.
- Set **Filename Prefix** (default is `wan22`) if you want different prefixes per tab.
- Use the **RIFE / frame interpolation** section only when you actually need interpolation; it increases compute.

If the inventory is wired, WAN model directories may appear in the dropdowns; otherwise, the text inputs accept raw paths.

## Starting a video run

- Use the **txt2vid** area to start from a text prompt, or **img2vid** when you want to motionize an input image.
- Check:
  - Resolution and FPS: higher values cost more VRAM and time.
  - Duration / frame count: long clips grow quickly in size.
- Click **Generate** in the tab header to start the run.

Generated videos appear in the results panel with metadata. Use different tabs for different presets (e.g., “short previews” vs. “final renders”) so you can reuse settings later.

