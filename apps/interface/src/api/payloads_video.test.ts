import { describe, expect, it } from 'vitest'

import { buildWanImg2VidPayload, buildWanTxt2VidPayload, buildWanVid2VidPayload } from './payloads_video'

describe('WAN video payload builders', () => {
  it('builds a txt2vid payload with stage overrides and video options', () => {
    const payload = buildWanTxt2VidPayload({
      device: 'CUDA',
      prompt: '  test prompt  ',
      negativePrompt: 'neg',
      width: 768,
      height: 432,
      fps: 24,
      frames: 16,
      high: {
        modelDir: '/models/wan22',
        sampler: 'Euler a',
        scheduler: 'Automatic',
        steps: 4,
        cfgScale: 7,
        seed: 42,
        flowShift: 8.2,
      },
      low: {
        modelDir: '/models/wan22',
        sampler: 'Euler',
        scheduler: 'Automatic',
        steps: 2,
        cfgScale: 5,
        seed: -1,
      },
      format: 'gguf',
      assets: {
        metadataDir: '/meta',
        textEncoder: 'wan22//abs/path/to/tenc.safetensors',
        vaePath: '/vae.safetensors',
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
    expect(payload.wan_high).toMatchObject({ model_dir: '/models/wan22', steps: 4, cfg_scale: 7, flow_shift: 8.2 })
    expect(payload.wan_low).toMatchObject({ model_dir: '/models/wan22', steps: 2, cfg_scale: 5 })
    expect(payload.video_interpolation).toMatchObject({ enabled: true, model: 'rife47.pth', times: 2 })
    expect(payload.wan_text_encoder_path).toBe('/abs/path/to/tenc.safetensors')
    expect(payload.wan_vae_path).toBe('/vae.safetensors')
    expect(payload.wan_metadata_dir).toBe('/meta')
  })

  it('builds an img2vid payload and emits wan_text_encoder_dir for directory inputs', () => {
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
        modelDir: '/models/wan22',
        sampler: '',
        scheduler: '',
        steps: 12,
        cfgScale: 7,
        seed: -1,
      },
      low: {
        modelDir: '/models/wan22',
        sampler: '',
        scheduler: '',
        steps: 12,
        cfgScale: 7,
        seed: -1,
      },
      format: 'auto',
      assets: {
        metadataDir: '',
        textEncoder: '/models/wan22-tenc',
        vaePath: '',
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
    expect(payload.wan_text_encoder_dir).toBe('/models/wan22-tenc')
    expect(payload.wan_text_encoder_path).toBeUndefined()
  })

  it('builds a vid2vid payload (multipart upload path optional) with flow settings', () => {
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
      high: { modelDir: '/models/wan22', sampler: '', scheduler: '', steps: 12, cfgScale: 7, seed: 1 },
      low: { modelDir: '/models/wan22', sampler: '', scheduler: '', steps: 12, cfgScale: 7, seed: 1 },
      format: 'gguf',
      assets: { metadataDir: '', textEncoder: '', vaePath: '' },
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
  })
})
