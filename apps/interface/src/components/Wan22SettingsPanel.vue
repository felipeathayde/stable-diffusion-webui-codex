<template>
  <div class="panel wan22-panel">
    <div class="panel-header">WAN 2.2 Settings</div>
    <div class="panel-body wan22-body">
      <div class="panel-section wan22-section">
        <h3 class="label-muted">Video Output</h3>
        <div class="wan22-grid">
          <div>
            <label class="label-muted">Filename Prefix</label>
            <input class="ui-input" type="text" v-model="store.filenamePrefix" placeholder="wan22" />
          </div>
          <div>
            <label class="label-muted">Format</label>
            <select class="select-md" v-model="store.videoFormat">
              <option v-for="opt in formatOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div>
            <label class="label-muted">Pixel Format</label>
            <select class="select-md" v-model="store.videoPixFormat">
              <option v-for="opt in pixFmtOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div>
            <label class="label-muted">CRF</label>
            <input class="ui-input" type="number" min="0" max="51" v-model.number="store.videoCrf" />
          </div>
          <div>
            <label class="label-muted">Loop Count</label>
            <input class="ui-input" type="number" min="0" v-model.number="store.videoLoopCount" />
          </div>
        </div>
        <div class="wan22-toggle-row">
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoPingpong" />
            Ping-pong playback
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoSaveMetadata" />
            Save metadata
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoSaveOutput" />
            Save output file
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoTrimToAudio" />
            Trim to audio
          </label>
        </div>
      </div>

      <div class="panel-section wan22-section">
        <div class="wan22-toggle-head">
          <h3 class="label-muted">Interpolation (RIFE)</h3>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.rifeEnabled" />
            Enable
          </label>
        </div>
        <div v-if="store.rifeEnabled" class="wan22-grid">
          <div>
            <label class="label-muted">Model</label>
            <input class="ui-input" type="text" v-model="store.rifeModel" />
          </div>
          <div>
            <label class="label-muted">Times</label>
            <input class="ui-input" type="number" min="1" max="8" step="1" v-model.number="store.rifeTimes" />
          </div>
        </div>
        <p v-else class="caption">Interpolation disabled.</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{ store: any }>()
const store = props.store

const formatOptions = ['video/h264-mp4', 'video/h265-mp4', 'video/gif']
const pixFmtOptions = ['yuv420p', 'yuv444p', 'yuv422p']
</script>
