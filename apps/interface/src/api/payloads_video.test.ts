/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for WAN video payload builders (txt2vid/img2vid/vid2vid).
Ensures request inputs (stage overrides + assets by sha) are mapped into the expected backend payload fields.

Symbols (top-level; keep in sync; no ghosts):
- `payloads_video.test` (module): WAN video payload builder tests (field mapping + defaults).
*/

import { describe, expect, it } from 'vitest'

import { buildWanImg2VidPayload, buildWanTxt2VidPayload, buildWanVid2VidPayload } from './payloads_video'

describe('WAN video payload builders', () => {
  it('builds a txt2vid payload with stage overrides and video options', () => {
    const hiSha = 'a'.repeat(64)
    const loSha = 'b'.repeat(64)
    const vaeSha = 'c'.repeat(64)
    const tencSha = 'd'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'

    const payload = buildWanTxt2VidPayload({
      device: 'CUDA',
      prompt: '  test prompt  ',
      negativePrompt: 'neg',
      width: 768,
      height: 432,
      fps: 24,
      frames: 16,
      high: {
        modelSha: hiSha,
        sampler: 'euler a',
        scheduler: 'simple',
        steps: 4,
        cfgScale: 7,
        seed: 42,
        flowShift: 8.2,
      },
      low: {
        modelSha: loSha,
        sampler: 'euler',
        scheduler: 'simple',
        steps: 2,
        cfgScale: 5,
        seed: -1,
      },
      format: 'gguf',
      assets: {
        metadataRepo: metaRepo,
        textEncoderSha: tencSha,
        vaeSha: vaeSha,
      },
      output: {
        filenamePrefix: 'wan22',
        format: 'video/h264-mp4',
        pixFmt: 'yuv420p',
        crf: 15,
        loopCount: 0,
        pingpong: false,
        trimToAudio: false,
        saveMetadata: true,
        saveOutput: true,
      },
      interpolation: {
        enabled: true,
        model: 'rife47.pth',
        times: 2,
      },
    })

    expect(payload.codex_device).toBe('cuda')
    expect(payload.txt2vid_prompt).toBe('test prompt')
    expect(payload.txt2vid_steps).toBe(4)
    expect(payload.txt2vid_cfg_scale).toBe(7)
    expect(payload.wan_format).toBe('gguf')
    expect(payload.wan_high).toMatchObject({ model_sha: hiSha, steps: 4, cfg_scale: 7, flow_shift: 8.2 })
    expect(payload.wan_low).toMatchObject({ model_sha: loSha, steps: 2, cfg_scale: 5 })
    expect(payload.video_interpolation).toMatchObject({ enabled: true, model: 'rife47.pth', times: 2 })
    expect(payload.wan_tenc_sha).toBe(tencSha)
    expect(payload.wan_vae_sha).toBe(vaeSha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
  })

  it('builds an img2vid payload with sha-selected assets', () => {
    const sha = 'e'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    const payload = buildWanImg2VidPayload({
      device: 'cpu',
      prompt: 'p',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 16,
      initImageData: 'data:image/png;base64,AAAA',
      high: {
        modelSha: sha,
        sampler: '',
        scheduler: '',
        steps: 12,
        cfgScale: 7,
        seed: -1,
      },
      low: {
        modelSha: sha,
        sampler: '',
        scheduler: '',
        steps: 12,
        cfgScale: 7,
        seed: -1,
      },
      format: 'auto',
      assets: {
        metadataRepo: metaRepo,
        textEncoderSha: sha,
        vaeSha: sha,
      },
      output: {
        filenamePrefix: '',
        format: '',
        pixFmt: '',
        crf: 15,
        loopCount: 0,
        pingpong: false,
        trimToAudio: false,
        saveMetadata: true,
        saveOutput: true,
      },
      interpolation: {
        enabled: false,
        model: '',
        times: 2,
      },
    })

    expect(payload.codex_device).toBe('cpu')
    expect(payload.img2vid_init_image).toBe('data:image/png;base64,AAAA')
    expect(payload.wan_tenc_sha).toBe(sha)
    expect(payload.wan_vae_sha).toBe(sha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
  })

  it('builds a vid2vid payload (multipart upload path optional) with flow settings', () => {
    const sha = 'f'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    const payload = buildWanVid2VidPayload({
      device: 'cuda',
      prompt: '  v2v  ',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 96,
      strength: 0.75,
      method: 'flow_chunks',
      useSourceFps: true,
      useSourceFrames: true,
      flowEnabled: true,
      flowUseLarge: false,
      flowDownscale: 2,
      high: { modelSha: sha, sampler: '', scheduler: '', steps: 12, cfgScale: 7, seed: 1 },
      low: { modelSha: sha, sampler: '', scheduler: '', steps: 12, cfgScale: 7, seed: 1 },
      format: 'gguf',
      assets: { metadataRepo: metaRepo, textEncoderSha: sha, vaeSha: sha },
      output: {
        filenamePrefix: 'wan22',
        format: 'video/h264-mp4',
        pixFmt: 'yuv420p',
        crf: 15,
        loopCount: 0,
        pingpong: false,
        trimToAudio: false,
        saveMetadata: true,
        saveOutput: true,
      },
      interpolation: { enabled: false, model: '', times: 2 },
    })

    expect(payload.codex_device).toBe('cuda')
    expect(payload.vid2vid_prompt).toBe('v2v')
    expect(payload.vid2vid_strength).toBe(0.75)
    expect(payload.vid2vid_method).toBe('flow_chunks')
    expect(payload.vid2vid_flow_enabled).toBe(true)
    expect(payload.wan_tenc_sha).toBe(sha)
    expect(payload.wan_vae_sha).toBe(sha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
  })
})
