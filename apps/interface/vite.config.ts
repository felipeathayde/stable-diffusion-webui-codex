// tags: vite-config, frontend-build
import { defineConfig, loadEnv } from 'vite'
import path from 'node:path'
import vue from '@vitejs/plugin-vue'
import tailwind from '@tailwindcss/vite'
import type { UserConfig as VitestUserConfig } from 'vitest'

// Restart vite dev server if root .env or Tailwind/PostCSS configs change
const watchRootConfigs = () => ({
  name: 'watch-root-configs',
  configureServer(server) {
    const root = path.resolve(__dirname, '../../')
    const globs = [
      path.join(root, '.env'),
      path.join(root, '.env.local'),
      path.join(__dirname, 'tailwind.config.*'),
      path.join(root, 'tailwind.config.*'),
      path.join(__dirname, 'postcss.config.*')
    ]
    server.watcher.add(globs)
    server.watcher.on('change', (file) => {
      const base = path.basename(file)
      if (base.startsWith('.env') || base.startsWith('tailwind.config') || base.startsWith('postcss.config')) {
        console.log(`[vite] Detected config change (${base}). Restarting dev server...`)
        server.restart()
      }
    })
  }
})

export default defineConfig(({ mode }) => {
  const repoRoot = path.resolve(__dirname, '../../')
  const env = loadEnv(mode, repoRoot, '')

  const WEB_PORT = Number(env.WEB_PORT || process.env.WEB_PORT || 7860)
  const SERVER_HOST = String(env.SERVER_HOST || process.env.SERVER_HOST || 'localhost')
  const API_HOST = String(env.API_HOST || process.env.API_HOST || 'localhost')
  // If API_PORT not explicitly provided, derive from UI port by -10 (7860→7850, 17860→17850, 27860→27850)
  let API_PORT = Number(env.API_PORT || process.env.API_PORT || (WEB_PORT - 10))
  const HMR_HOST = String(env.HMR_HOST || process.env.HMR_HOST || SERVER_HOST)
  const HMR_PROTOCOL = String(env.HMR_PROTOCOL || process.env.HMR_PROTOCOL || 'ws')

  const allowedHosts = new Set<string>(['localhost', '127.0.0.1', '::1', 'webui.sangoi.dev'])
  ;(env.ALLOWED_HOSTS || process.env.ALLOWED_HOSTS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .forEach((h) => allowedHosts.add(h))

  const vitestTestConfig: VitestUserConfig['test'] = {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}', 'src/**/*.spec.{ts,tsx}'],
  }

  return {
    envDir: repoRoot,
    plugins: [vue(), tailwind(), watchRootConfigs()],
    server: {
      port: WEB_PORT,
      strictPort: true,
      host: SERVER_HOST,
      allowedHosts: Array.from(allowedHosts),
      proxy: {
        '/api': {
          target: `http://${API_HOST}:${API_PORT}`,
          changeOrigin: true
        }
      },
      hmr: {
        host: HMR_HOST,
        port: WEB_PORT,
        protocol: HMR_PROTOCOL as 'ws' | 'wss'
      }
    },
    test: vitestTestConfig,
  }
})
