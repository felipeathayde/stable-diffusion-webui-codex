<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: In-tab dependency status panel with backend-driven OK/ERROR rows.
Renders backend dependency checks with binary indicators and actionable messages so tabs can expose readiness and disable generation
without frontend guesswork.

Symbols (top-level; keep in sync; no ghosts):
- `DependencyCheckPanel` (component): Displays backend dependency checks for one semantic engine.
-->

<template>
  <section class="panel dependency-check-panel">
    <div class="panel-header">
      <span>{{ title }}</span>
      <span v-if="status" :class="['dependency-overall', status.ready ? 'is-ready' : 'is-error']">
        {{ status.ready ? 'OK' : 'ERROR' }}
      </span>
    </div>
    <div class="panel-body">
      <p v-if="!status" class="caption">
        Dependency checks are unavailable for {{ engineLabel }}.
      </p>
      <ul v-else class="dependency-check-list">
        <li v-for="row in status.checks" :key="row.id" class="dependency-check-row">
          <div class="dependency-check-text">
            <div class="dependency-check-label">{{ row.label }}</div>
            <p class="dependency-check-message">{{ row.message }}</p>
          </div>
          <span
            :class="['dependency-light', row.ok ? 'is-ready' : 'is-error']"
            :title="row.ok ? 'Check OK' : 'Check failed'"
            :aria-label="row.ok ? 'Check OK' : 'Check failed'"
          />
        </li>
      </ul>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { EngineDependencyStatus } from '../api/types'

withDefaults(defineProps<{
  status: EngineDependencyStatus | null
  title?: string
  engineLabel?: string
}>(), {
  title: 'Dependency Check',
  engineLabel: 'this engine',
})
</script>

<style scoped>
.dependency-check-panel {
  border: 1px solid rgba(138, 148, 168, 0.24);
  background:
    radial-gradient(circle at 8% 10%, rgba(27, 45, 66, 0.45) 0, rgba(27, 45, 66, 0) 44%),
    linear-gradient(180deg, rgba(14, 20, 30, 0.94), rgba(10, 16, 24, 0.96));
}

.dependency-overall {
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  font-weight: 700;
}

.dependency-overall.is-ready {
  background: rgba(42, 122, 79, 0.26);
  color: #86e7b9;
}

.dependency-overall.is-error {
  background: rgba(142, 47, 58, 0.28);
  color: #ff9aa1;
}

.dependency-check-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.dependency-check-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.55rem 0.7rem;
  border-radius: 0.55rem;
  border: 1px solid rgba(120, 133, 154, 0.24);
  background: rgba(18, 26, 38, 0.72);
}

.dependency-check-text {
  min-width: 0;
}

.dependency-check-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: rgba(236, 241, 249, 0.95);
}

.dependency-check-message {
  margin: 0.18rem 0 0;
  color: rgba(169, 184, 206, 0.94);
  font-size: 0.8rem;
  line-height: 1.35;
}

.dependency-light {
  flex: 0 0 auto;
  width: 0.95rem;
  height: 0.95rem;
  border-radius: 999px;
  border: 1px solid rgba(196, 205, 219, 0.36);
}

.dependency-light.is-ready {
  background: radial-gradient(circle at 33% 33%, #d5ffe9 0, #3ad17f 44%, #1f8a53 100%);
  box-shadow: 0 0 0.65rem rgba(70, 212, 133, 0.46);
}

.dependency-light.is-error {
  background: radial-gradient(circle at 33% 33%, #ffd9dc 0, #ff6a6a 42%, #be3333 100%);
  box-shadow: 0 0 0.65rem rgba(236, 86, 86, 0.5);
}

@media (max-width: 768px) {
  .dependency-check-row {
    padding: 0.5rem 0.58rem;
    gap: 0.7rem;
  }
}
</style>
