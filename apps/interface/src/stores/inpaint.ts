/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Inpaint/init-image store for the frontend.
Keeps an init image (data URL + name) in state and exposes helpers to set/clear it from an uploaded file.

Symbols (top-level; keep in sync; no ghosts):
- `useInpaintStore` (store): Pinia store for init-image state used by inpaint/img2img flows.
- `readFileAsDataURL` (function): Reads a `File` into a `data:` URL via `FileReader`.
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useInpaintStore = defineStore('inpaint', () => {
  const initImageData = ref<string>('')
  const initImageName = ref<string>('')
  const initImagePreview = computed(() => initImageData.value || '')
  const hasInitImage = computed(() => Boolean(initImageData.value))

  async function setInitImage(file: File): Promise<void> {
    initImageName.value = file.name
    const dataUrl = await readFileAsDataURL(file)
    initImageData.value = dataUrl
  }

  async function clearInitImage(): Promise<void> {
    initImageData.value = ''
    initImageName.value = ''
  }

  return {
    initImageData,
    initImageName,
    initImagePreview,
    hasInitImage,
    setInitImage,
    clearInitImage,
  }
})

async function readFileAsDataURL(file: File): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}
