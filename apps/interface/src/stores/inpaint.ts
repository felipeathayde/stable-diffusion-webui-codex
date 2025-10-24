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

