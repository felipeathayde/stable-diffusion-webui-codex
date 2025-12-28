import { onBeforeUnmount, ref } from 'vue'

export function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  textarea.remove()
}

export function useResultsCard(options: { noticeDurationMs?: number } = {}) {
  const notice = ref('')
  let noticeTimer: number | null = null

  const noticeDurationMs = Number.isFinite(options.noticeDurationMs)
    ? Math.max(0, Number(options.noticeDurationMs))
    : 2000

  function clearNotice(): void {
    notice.value = ''
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
    noticeTimer = null
  }

  function toast(message: string): void {
    notice.value = message
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
    noticeTimer = window.setTimeout(() => {
      notice.value = ''
      noticeTimer = null
    }, noticeDurationMs)
  }

  async function copyText(text: string, successMessage = 'Copied to clipboard.'): Promise<void> {
    try {
      await copyToClipboard(text)
      toast(successMessage)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err))
    }
  }

  async function copyJson(value: unknown, successMessage = 'Copied JSON.'): Promise<void> {
    try {
      await copyText(JSON.stringify(value, null, 2), successMessage)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err))
    }
  }

  onBeforeUnmount(() => {
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
  })

  return {
    notice,
    toast,
    clearNotice,
    copyText,
    copyJson,
    formatJson,
  }
}
