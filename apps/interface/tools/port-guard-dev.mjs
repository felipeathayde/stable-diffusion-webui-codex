#!/usr/bin/env node
import net from 'node:net'
import { spawn } from 'node:child_process'
import path from 'node:path'

const COLOR = {
  red: (s) => `\u001b[31m${s}\u001b[0m`,
  yellow: (s) => `\u001b[33m${s}\u001b[0m`,
  cyan: (s) => `\u001b[36m${s}\u001b[0m`,
}

function checkPort(port, host = '0.0.0.0') {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.once('error', () => resolve(false))
    srv.once('listening', () => srv.close(() => resolve(true)))
    srv.listen(port, host)
  })
}

async function findPort(ranges) {
  for (const [start, end] of ranges) {
    for (let p = start; p <= end; p++) {
      // eslint-disable-next-line no-await-in-loop
      if (await checkPort(p)) return p
    }
  }
  return null
}

const cliArgs = process.argv.slice(2)

async function main() {
  const base = Number(process.env.WEB_PORT) || 7860
  const c1 = await checkPort(base)
  if (c1) return runVite(base, false, cliArgs)
  const f1 = base + 10000
  const c2 = await checkPort(f1)
  if (c2) {
    banner(f1, base)
    return runVite(f1, true, cliArgs)
  }
  const f2 = base + 20000
  const c3 = await checkPort(f2)
  if (c3) {
    banner(f2, base)
    return runVite(f2, true, cliArgs)
  }
  console.error(COLOR.red(`[port-guard] No free port for UI. Tried ${base}, ${f1}, ${f2}.`))
  process.exit(1)
}

function runVite(port, show, extraArgs = []) {
  process.env.WEB_PORT = String(port)
  const viteBin = path.join(process.cwd(), 'node_modules', '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite')
  const useShell = process.platform === 'win32'
  const child = spawn(viteBin, extraArgs, { stdio: 'inherit', env: process.env, shell: useShell })
  child.on('exit', (code) => process.exit(code ?? 0))
}

function banner(chosen, base) {
  const text = [
    '',
    '==============================================',
    '  PORT GUARD — UI Fallback                   ',
    '==============================================',
    ` Using UI port ${chosen} (base ${base} busy).`,
    '==============================================',
    ''
  ].join('\n')
  console.log(COLOR.cyan(text))
}

main().catch((err) => {
  console.error(COLOR.red(`[port-guard] ${err?.stack || err}`))
  process.exit(1)
})
