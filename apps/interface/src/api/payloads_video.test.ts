/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for WAN video payload builders (txt2vid/img2vid/vid2vid).
Ensures request inputs (stage overrides + assets by sha) are mapped into the expected backend payload fields, including
WAN dimension snapping to `%16 == 0` (rounded up; Diffusers parity), `settings_revision` propagation, and scheduler-override omission.

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
      settingsRevision: 5,
      prompt: '  test prompt  ',
      negativePrompt: 'neg',
      width: 768,
      height: 432,
      fps: 24,
      frames: 17,
      attentionMode: 'global',
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
        returnFrames: true,
      },
      interpolation: {
        enabled: true,
        model: 'rife47.pth',
        times: 2,
      },
    })

    expect(payload.codex_device).toBe('cuda')
    expect(payload.settings_revision).toBe(5)
    expect(payload.txt2vid_prompt).toBe('test prompt')
    expect(payload.txt2vid_steps).toBe(6)
    expect(payload.txt2vid_cfg_scale).toBe(7)
    expect(payload.wan_format).toBe('gguf')
    expect(payload.wan_high).toMatchObject({ model_sha: hiSha, steps: 4, cfg_scale: 7, flow_shift: 8.2 })
    expect(payload.wan_low).toMatchObject({ model_sha: loSha, steps: 2, cfg_scale: 5 })
    expect(payload.video_interpolation).toMatchObject({ enabled: true, model: 'rife47.pth', times: 2 })
    expect(payload.wan_tenc_sha).toBe(tencSha)
    expect(payload.wan_vae_sha).toBe(vaeSha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
    expect(payload.video_return_frames).toBe(true)
    expect(payload).not.toHaveProperty('txt2vid_scheduler')
    expect(payload.wan_high).not.toHaveProperty('scheduler')
    expect(payload.wan_low).not.toHaveProperty('scheduler')
  })

  it('builds an img2vid payload with sha-selected assets', () => {
    const sha = 'e'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    const payload = buildWanImg2VidPayload({
      device: 'cpu',
      settingsRevision: 11,
      prompt: 'p',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 17,
      attentionMode: 'sliding',
      initImageData: 'data:image/png;base64,AAAA',
      chunkFrames: 37,
      overlapFrames: 8,
      anchorAlpha: 0.35,
      chunkSeedMode: 'increment',
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
    expect(payload.settings_revision).toBe(11)
    expect(payload.img2vid_init_image).toBe('data:image/png;base64,AAAA')
    expect(payload.img2vid_steps).toBe(24)
    expect(payload.wan_tenc_sha).toBe(sha)
    expect(payload.wan_vae_sha).toBe(sha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
    expect(payload.gguf_attention_mode).toBe('sliding')
    expect(payload.img2vid_chunk_frames).toBe(37)
    expect(payload.img2vid_overlap_frames).toBe(8)
    expect(payload.img2vid_anchor_alpha).toBe(0.35)
    expect(payload.img2vid_chunk_seed_mode).toBe('increment')
    expect(payload).not.toHaveProperty('img2vid_scheduler')
  })

  it('omits img2vid chunk-only fields when chunkFrames is disabled', () => {
    const sha = '9'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    const payload = buildWanImg2VidPayload({
      device: 'cpu',
      settingsRevision: 17,
      prompt: 'p',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 17,
      attentionMode: 'global',
      initImageData: 'data:image/png;base64,AAAA',
      chunkFrames: 0,
      overlapFrames: 6,
      anchorAlpha: 0.6,
      chunkSeedMode: 'random',
      high: {
        modelSha: sha,
        sampler: '',
        scheduler: '',
        steps: 8,
        cfgScale: 6,
        seed: -1,
      },
      low: {
        modelSha: sha,
        sampler: '',
        scheduler: '',
        steps: 8,
        cfgScale: 6,
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

    expect(payload).not.toHaveProperty('img2vid_chunk_frames')
    expect(payload).not.toHaveProperty('img2vid_overlap_frames')
    expect(payload).not.toHaveProperty('img2vid_anchor_alpha')
    expect(payload).not.toHaveProperty('img2vid_chunk_seed_mode')
  })

  it('builds a vid2vid payload (multipart upload path optional) with flow settings', () => {
    const sha = 'f'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    const payload = buildWanVid2VidPayload({
      device: 'cuda',
      settingsRevision: 13,
      prompt: '  v2v  ',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 97,
      attentionMode: 'global',
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
    expect(payload.settings_revision).toBe(13)
    expect(payload.vid2vid_prompt).toBe('v2v')
    expect(payload.vid2vid_steps).toBe(24)
    expect(payload.vid2vid_strength).toBe(0.75)
    expect(payload.vid2vid_method).toBe('flow_chunks')
    expect(payload.vid2vid_flow_enabled).toBe(true)
    expect(payload.wan_tenc_sha).toBe(sha)
    expect(payload.wan_vae_sha).toBe(sha)
    expect(payload.wan_metadata_repo).toBe(metaRepo)
    expect(payload).not.toHaveProperty('vid2vid_scheduler')
  })

  it('rounds WAN video dimensions up to a multiple of 16', () => {
    const sha = '1'.repeat(64)
    const metaRepo = 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
    const payload = buildWanTxt2VidPayload({
      device: 'cuda',
      settingsRevision: 2,
      prompt: 'p',
      negativePrompt: '',
      width: 480,
      height: 360,
      fps: 24,
      frames: 17,
      attentionMode: 'global',
      high: { modelSha: sha, sampler: '', scheduler: '', steps: 2, cfgScale: 1, seed: -1 },
      low: { modelSha: sha, sampler: '', scheduler: '', steps: 2, cfgScale: 1, seed: -1 },
      format: 'auto',
      assets: { metadataRepo: metaRepo, textEncoderSha: sha, vaeSha: sha },
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
      interpolation: { enabled: false, model: '', times: 2 },
    })

    expect(payload.txt2vid_width).toBe(480)
    expect(payload.txt2vid_height).toBe(368)
  })
})
