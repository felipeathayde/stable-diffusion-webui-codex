<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: XYZ sweep view.
Hosts the standalone `/xyz` route as a wrapper around the shared embedded XYZ card.

Symbols (top-level; keep in sync; no ghosts):
- `XyzPlot` (component): XYZ sweep route view component.
-->

<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">XYZ sweep</div>
      <div class="panel-body">
        <p class="caption">This route is kept for compatibility. XYZ is also available in Generation Parameters inside image model tabs.</p>
        <XyzSweepCard />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import XyzSweepCard from '../components/XyzSweepCard.vue'
</script>
