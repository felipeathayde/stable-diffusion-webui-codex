/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: UI consistency scan for frontend style-contract drift.
Generates a deterministic markdown report with static/dynamic style findings,
selector duplication signals, and docs/toolchain reference drift checks.
Supports strict mode (`--strict`) to fail loud on style-contract drift.

Symbols (top-level; keep in sync; no ghosts):
- `main` (function): Runs the report scan and writes summary output.
- `collectFiles` (function): Recursively collects files by extension.
- `scanStaticInlineStyles` (function): Finds literal `style="..."` in Vue templates.
- `scanDynamicStyleBindings` (function): Finds dynamic `:style`/`v-bind:style` bindings in Vue templates.
- `scanScopedStyles` (function): Finds scoped style blocks in Vue SFCs.
- `scanSelectorDuplicates` (function): Detects duplicated selectors across CSS files.
- `scanDocsToolchainDrift` (function): Reports stale docs references against actual toolchain.
- `parseFlags` (function): Parses CLI flags for strict/fail-loud execution mode.
*/

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const INTERFACE_ROOT = path.resolve(SCRIPT_DIR, '..')
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..', '..')
const SRC_ROOT = path.join(INTERFACE_ROOT, 'src')
const STYLES_ROOT = path.join(SRC_ROOT, 'styles')
const REPORT_DIR = path.join(INTERFACE_ROOT, '.reports')
const REPORT_PATH = path.join(REPORT_DIR, 'ui-consistency-report.md')

const STYLE_GUIDE_PATH = path.join(REPO_ROOT, '.sangoi', 'frontend', 'guidelines', 'frontend-style-guide.md')
const ARCH_GUIDE_PATH = path.join(REPO_ROOT, '.sangoi', 'frontend', 'guidelines', 'frontend-architecture-guide.md')
const PACKAGE_JSON_PATH = path.join(INTERFACE_ROOT, 'package.json')

const SELECTOR_WATCHLIST = ['.viewer-card', '.viewer-empty', '.panel-stack', '.form-row']
const DYNAMIC_STYLE_ALLOWLIST = new Set([
  'batchMenuStyle',
  'zoomStyle',
  'previewStyle',
  'contentTransformStyle',
  'brushCursorStyle',
  'props.bodyStyle',
  'guidedTooltipStyle',
])

function collectFiles(root, extensions) {
  const out = []
  if (!fs.existsSync(root)) return out
  const stack = [root]
  while (stack.length > 0) {
    const current = stack.pop()
    if (!current) continue
    const entries = fs.readdirSync(current, { withFileTypes: true })
    for (const entry of entries) {
      const abs = path.join(current, entry.name)
      if (entry.isDirectory()) {
        stack.push(abs)
        continue
      }
      if (!entry.isFile()) continue
      if (extensions.some((extension) => entry.name.endsWith(extension))) {
        out.push(abs)
      }
    }
  }
  return out.sort((left, right) => left.localeCompare(right))
}

function toRel(absPath) {
  return path.relative(REPO_ROOT, absPath).replaceAll(path.sep, '/')
}

function scanStaticInlineStyles(vueFiles) {
  const findings = []
  for (const filePath of vueFiles) {
    const content = fs.readFileSync(filePath, 'utf-8')
    const lines = content.split('\n')
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]
      if (!line.includes('style=')) continue
      if (line.includes(':style=') || line.includes('v-bind:style=')) continue
      if (!line.includes('style="') && !line.includes("style='")) continue
      findings.push({
        file: toRel(filePath),
        line: index + 1,
        snippet: line.trim(),
      })
    }
  }
  return findings
}

function scanDynamicStyleBindings(vueFiles) {
  const findings = []
  const regex = /(?:\:style|v-bind:style)\s*=\s*("([^"]+)"|'([^']+)')/g
  for (const filePath of vueFiles) {
    const content = fs.readFileSync(filePath, 'utf-8')
    const lines = content.split('\n')
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]
      if (!line.includes(':style=') && !line.includes('v-bind:style=')) continue
      regex.lastIndex = 0
      let match = regex.exec(line)
      while (match) {
        const expression = String(match[2] || match[3] || '').trim()
        const isAllowed = DYNAMIC_STYLE_ALLOWLIST.has(expression)
        findings.push({
          file: toRel(filePath),
          line: index + 1,
          expression,
          allowed: isAllowed,
          snippet: line.trim(),
        })
        match = regex.exec(line)
      }
    }
  }
  return findings
}

function scanScopedStyles(vueFiles) {
  const findings = []
  const scopedRegex = /<style\b[^>]*\bscoped\b[^>]*>/
  for (const filePath of vueFiles) {
    const content = fs.readFileSync(filePath, 'utf-8')
    const lines = content.split('\n')
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]
      if (!scopedRegex.test(line)) continue
      findings.push({
        file: toRel(filePath),
        line: index + 1,
        snippet: line.trim(),
      })
    }
  }
  return findings
}

function extractSelectors(cssContent) {
  const selectors = []
  const lines = cssContent.split('\n')
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (!line.includes('{')) continue
    if (line.trimStart().startsWith('@')) continue
    const beforeBrace = line.split('{', 1)[0]
    if (!beforeBrace.includes('.')) continue
    const candidates = beforeBrace.split(',').map((part) => part.trim()).filter(Boolean)
    for (const candidate of candidates) {
      if (!candidate.startsWith('.')) continue
      selectors.push({ selector: candidate, line: index + 1 })
    }
  }
  return selectors
}

function scanSelectorDuplicates(cssFiles) {
  const selectorMap = new Map()
  for (const filePath of cssFiles) {
    const relPath = toRel(filePath)
    if (relPath.includes('/EXAMPLE-')) continue
    const selectors = extractSelectors(fs.readFileSync(filePath, 'utf-8'))
    for (const entry of selectors) {
      const list = selectorMap.get(entry.selector) ?? []
      list.push({ file: relPath, line: entry.line })
      selectorMap.set(entry.selector, list)
    }
  }

  const duplicates = []
  for (const [selector, locations] of selectorMap.entries()) {
    const uniqueFiles = [...new Set(locations.map((location) => location.file))]
    if (uniqueFiles.length <= 1) continue
    duplicates.push({ selector, locations, fileCount: uniqueFiles.length })
  }

  duplicates.sort((left, right) => {
    if (right.fileCount !== left.fileCount) return right.fileCount - left.fileCount
    return left.selector.localeCompare(right.selector)
  })

  const watchlistHits = duplicates.filter((entry) => SELECTOR_WATCHLIST.includes(entry.selector))
  return { duplicates, watchlistHits }
}

function readTextIfPresent(filePath) {
  if (!fs.existsSync(filePath)) return ''
  return fs.readFileSync(filePath, 'utf-8')
}

function scanDocsToolchainDrift() {
  const styleGuide = readTextIfPresent(STYLE_GUIDE_PATH)
  const architectureGuide = readTextIfPresent(ARCH_GUIDE_PATH)
  const packageJson = JSON.parse(readTextIfPresent(PACKAGE_JSON_PATH) || '{}')
  const scripts = packageJson.scripts ?? {}

  const checks = []

  if (styleGuide.includes('apps/interface/tailwind.config.ts')) {
    checks.push({
      key: 'tailwind.config.ts reference',
      ok: fs.existsSync(path.join(INTERFACE_ROOT, 'tailwind.config.ts')),
      expected: 'apps/interface/tailwind.config.ts exists',
    })
  }

  if (architectureGuide.includes('.legacy/root-archive')) {
    checks.push({
      key: '.legacy/root-archive reference',
      ok: fs.existsSync(path.join(REPO_ROOT, '.legacy', 'root-archive')),
      expected: '.legacy/root-archive exists',
    })
  }

  if (architectureGuide.includes('npm run build:ts')) {
    checks.push({
      key: 'build:ts script reference',
      ok: typeof scripts['build:ts'] === 'string',
      expected: 'package.json contains scripts.build:ts',
    })
  }

  if (architectureGuide.includes('npm run watch:ts')) {
    checks.push({
      key: 'watch:ts script reference',
      ok: typeof scripts['watch:ts'] === 'string',
      expected: 'package.json contains scripts.watch:ts',
    })
  }

  return checks
}

function renderFindings(title, findings, limit = 20) {
  if (findings.length === 0) return `## ${title}\n- none\n`
  const lines = [`## ${title}`, `- total: ${findings.length}`]
  for (const finding of findings.slice(0, limit)) {
    lines.push(`- ${finding.file}:${finding.line} — \`${finding.snippet}\``)
  }
  if (findings.length > limit) {
    lines.push(`- ... ${findings.length - limit} more`) 
  }
  return `${lines.join('\n')}\n`
}

function renderDynamicStyleSection(findings) {
  if (findings.length === 0) return '## Dynamic `:style` / `v-bind:style` Bindings\n- none\n'
  const disallowed = findings.filter((finding) => !finding.allowed)
  const lines = [
    '## Dynamic `:style` / `v-bind:style` Bindings',
    `- total: ${findings.length}`,
    `- allowlisted: ${findings.length - disallowed.length}`,
    `- disallowed: ${disallowed.length}`,
  ]
  for (const finding of findings.slice(0, 30)) {
    const status = finding.allowed ? 'allowlisted' : 'disallowed'
    lines.push(
      `- [${status}] ${finding.file}:${finding.line} — expr=\`${finding.expression}\` — \`${finding.snippet}\``,
    )
  }
  if (findings.length > 30) {
    lines.push(`- ... ${findings.length - 30} more`)
  }
  return `${lines.join('\n')}\n`
}

function renderDuplicateSection(result) {
  const { duplicates, watchlistHits } = result
  if (duplicates.length === 0) return '## Duplicated Selectors Across Files\n- none\n'

  const lines = [
    '## Duplicated Selectors Across Files',
    `- total duplicated selectors: ${duplicates.length}`,
    `- watchlist hits: ${watchlistHits.length}`,
  ]

  for (const entry of watchlistHits) {
    const first = entry.locations.slice(0, 4).map((location) => `${location.file}:${location.line}`).join(', ')
    lines.push(`- [watchlist] ${entry.selector} -> ${entry.fileCount} files (${first})`)
  }

  for (const entry of duplicates.slice(0, 20)) {
    if (SELECTOR_WATCHLIST.includes(entry.selector)) continue
    const first = entry.locations.slice(0, 3).map((location) => `${location.file}:${location.line}`).join(', ')
    lines.push(`- ${entry.selector} -> ${entry.fileCount} files (${first})`)
  }
  return `${lines.join('\n')}\n`
}

function renderDocsDrift(checks) {
  if (checks.length === 0) return '## Docs/Toolchain Drift\n- no tracked reference checks were detected\n'
  const lines = ['## Docs/Toolchain Drift']
  for (const check of checks) {
    const status = check.ok ? 'OK' : 'DRIFT'
    lines.push(`- [${status}] ${check.key} — expected: ${check.expected}`)
  }
  return `${lines.join('\n')}\n`
}

function renderSmokeSet() {
  return [
    '## Backend Malformed-File Smoke Command Set (documented, not executed by this script)',
    '- `cp apps/interface/tabs.json /tmp/tabs.json.bak && printf "{\\"oops\\":1}" > apps/interface/tabs.json`',
    '- `cp apps/interface/workflows.json /tmp/workflows.json.bak && printf "{\\"oops\\":1}" > apps/interface/workflows.json`',
    '- `cp apps/interface/presets.json /tmp/presets.json.bak && printf "{\\"oops\\":1}" > apps/interface/presets.json`',
    '- `curl -i http://127.0.0.1:7850/api/ui/tabs` (expected fail-loud)',
    '- `curl -i http://127.0.0.1:7850/api/ui/workflows` (expected fail-loud)',
    '- `curl -i http://127.0.0.1:7850/api/ui/presets` (expected fail-loud)',
    '- restore backups from `/tmp/*.bak` after smoke run',
    '',
  ].join('\n')
}

function parseFlags(argv) {
  const strict = argv.includes('--strict') || argv.includes('--fail-on-drift')
  return { strict }
}

function main() {
  const flags = parseFlags(process.argv.slice(2))
  const vueFiles = collectFiles(SRC_ROOT, ['.vue'])
  const cssFiles = [
    path.join(SRC_ROOT, 'styles.css'),
    ...collectFiles(STYLES_ROOT, ['.css']),
  ].filter((filePath, index, array) => fs.existsSync(filePath) && array.indexOf(filePath) === index)

  const inlineFindings = scanStaticInlineStyles(vueFiles)
  const dynamicStyleFindings = scanDynamicStyleBindings(vueFiles)
  const scopedFindings = scanScopedStyles(vueFiles)
  const duplicateResult = scanSelectorDuplicates(cssFiles)
  const docsChecks = scanDocsToolchainDrift()
  const disallowedDynamicStyleFindings = dynamicStyleFindings.filter((finding) => !finding.allowed)

  const generatedAt = new Date().toISOString()
  const report = [
    '# UI Consistency Report',
    `- generated_at: ${generatedAt}`,
    `- source_root: ${toRel(SRC_ROOT)}`,
    `- css_root: ${toRel(STYLES_ROOT)}`,
    `- strict_mode: ${flags.strict ? 'enabled' : 'disabled'}`,
    '',
    renderFindings('Static Inline Styles (`style="..."`)', inlineFindings),
    renderDynamicStyleSection(dynamicStyleFindings),
    renderFindings('Scoped `<style>` Blocks', scopedFindings),
    renderDuplicateSection(duplicateResult),
    renderDocsDrift(docsChecks),
    renderSmokeSet(),
  ].join('\n')

  fs.mkdirSync(REPORT_DIR, { recursive: true })
  fs.writeFileSync(REPORT_PATH, report, 'utf-8')

  const driftCount = docsChecks.filter((check) => !check.ok).length
  const styleDriftCount =
    inlineFindings.length +
    scopedFindings.length +
    disallowedDynamicStyleFindings.length +
    duplicateResult.duplicates.length
  console.log(`[ui-consistency-report] wrote ${path.relative(INTERFACE_ROOT, REPORT_PATH)}`)
  console.log(
    `[ui-consistency-report] inline=${inlineFindings.length} dynamic=${dynamicStyleFindings.length} dynamic_disallowed=${disallowedDynamicStyleFindings.length} scoped=${scopedFindings.length} duplicated_selectors=${duplicateResult.duplicates.length} docs_drift=${driftCount}`,
  )
  if (flags.strict) {
    if (styleDriftCount > 0) {
      console.error(
        `[ui-consistency-report] strict mode failed: style_drift=${styleDriftCount} (docs_drift=${driftCount} is non-gating)`,
      )
      process.exitCode = 1
      return
    }
    console.log(`[ui-consistency-report] strict mode passed: style_drift=${styleDriftCount}`)
    return
  }
  console.log('[ui-consistency-report] report mode: non-strict exit 0')
}

main()
