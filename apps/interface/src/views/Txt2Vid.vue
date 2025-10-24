<template>
  <section class="panels">
    <div class="panel-stack" ref="leftStack">
      <div class="panel">
        <div class="panel-header"><span>TEXT TO VIDEO</span>
          <div class="toolbar prompt-toolbar">
            <button class="btn btn-sm btn-secondary" type="button" @click="showCkpt=true">Checkpoints</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showTI=true">Textual Inversion</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showLora=true">LoRA</button>
            <label class="label-muted styles-label">Styles</label>
            <input class="ui-input styles-input" list="style-list" v-model="styleName" placeholder="Filter styles" />
            <datalist id="style-list"><option v-for="s in styleNames" :key="s" :value="s" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="showStyle=true">New</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyStyle(styleName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <PromptFields v-model:prompt="store.prompt" v-model:negative="store.negativePrompt" />

          <div v-if="store.status === 'running'" class="card text-sm">
            <p><strong>Stage:</strong> {{ store.progress.stage }}</p>
            <p v-if="store.progress.percent !== null">Progress: {{ store.progress.percent?.toFixed(1) }}%</p>
            <p v-if="store.progress.step !== null && store.progress.totalSteps !== null">
              Step {{ store.progress.step }} / {{ store.progress.totalSteps }}
            </p>
          </div>

          <div v-if="store.status === 'error'" class="card text-sm text-destructive">
            {{ store.errorMessage }}
          </div>
        </div>
      </div>

      <ParamBlocksRenderer v-if="uiBlocks.loaded" tab="txt2vid" :blocks="uiBlocks.blocks" :store="store" />
      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <template v-if="isWan22Active">
            <div class="panel-section">
              <h3 class="label-muted">High Noise</h3>
              <GenerationSettingsCard
                :samplers="store.samplers"
                :schedulers="store.schedulers"
                :sampler="store.hiSampler"
                :scheduler="store.hiScheduler"
                :steps="store.hiSteps"
                :width="store.width"
                :height="store.height"
                :cfg-scale="store.hiCfgScale"
                :seed="store.hiSeed"
                :batch-size="store.batchSize"
                :batch-count="1"
                @update:sampler="(v:string)=>store.hiSampler=v"
                @update:scheduler="(v:string)=>store.hiScheduler=v"
                @update:steps="(v:number)=>store.hiSteps=v"
                @update:width="(v:number)=>store.width=v"
                @update:height="(v:number)=>store.height=v"
                @update:cfgScale="(v:number)=>store.hiCfgScale=v"
                @update:seed="(v:number)=>store.hiSeed=v"
                @update:batchSize="(v:number)=>store.batchSize=v"
                @random-seed="store.randomizeSeedHi"
                @reuse-seed="store.reuseSeedHi"
              />
            </div>
            <div class="panel-section">
              <h3 class="label-muted">Low Noise</h3>
              <GenerationSettingsCard
                :samplers="store.samplers"
                :schedulers="store.schedulers"
                :sampler="store.loSampler"
                :scheduler="store.loScheduler"
                :steps="store.loSteps"
                :width="store.width"
                :height="store.height"
                :cfg-scale="store.loCfgScale"
                :seed="store.loSeed"
                :batch-size="store.batchSize"
                :batch-count="1"
                @update:sampler="(v:string)=>store.loSampler=v"
                @update:scheduler="(v:string)=>store.loScheduler=v"
                @update:steps="(v:number)=>store.loSteps=v"
                @update:width="(v:number)=>store.width=v"
                @update:height="(v:number)=>store.height=v"
                @update:cfgScale="(v:number)=>store.loCfgScale=v"
                @update:seed="(v:number)=>store.loSeed=v"
                @update:batchSize="(v:number)=>store.batchSize=v"
                @random-seed="store.randomizeSeedLo"
                @reuse-seed="store.reuseSeedLo"
              />
            </div>
          </template>
          <template v-else>
            <GenerationSettingsCard
              :samplers="store.samplers"
              :schedulers="store.schedulers"
              :sampler="store.sampler"
              :scheduler="store.scheduler"
              :steps="store.steps"
              :width="store.width"
              :height="store.height"
              :cfg-scale="store.cfgScale"
              :seed="store.seed"
              :batch-size="store.batchSize"
              :batch-count="1"
              @update:sampler="(v:string)=>store.sampler=v"
              @update:scheduler="(v:string)=>store.scheduler=v"
              @update:steps="(v:number)=>store.steps=v"
              @update:width="(v:number)=>store.width=v"
              @update:height="(v:number)=>store.height=v"
              @update:cfgScale="(v:number)=>store.cfgScale=v"
              @update:seed="(v:number)=>store.seed=v"
              @update:batchSize="(v:number)=>store.batchSize=v"
              @random-seed="store.randomizeSeed"
              @reuse-seed="store.reuseSeed"
            />
          </template>

          <VideoSettingsCard
            :frames="store.frames"
            :fps="store.fps"
            @update:frames="(v:number)=>store.frames=v"
            @update:fps="(v:number)=>store.fps=v"
          />
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky"><span>Results</span>
          <div class="header-center"><button class="btn btn-md btn-primary results-generate" :disabled="store.isRunning" @click="onGenerate">Generate</button></div>
          <div class="header-right results-actions">
            <input class="ui-input" :list="'preset-list'" v-model="presetName" placeholder="Preset" />
            <datalist id="preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <ResultViewer mode="video" :frames="store.framesResult" :toDataUrl="store.toDataUrl" emptyText="Generated frames will appear here." />
        </div>
      </div>
      <div class="panel" v-if="store.info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre>{{ asJson(store.info) }}</pre>
        </div>
      </div>
    </div>

    <!-- Modals -->
    <CheckpointModal v-model="showCkpt" />
    <LoraModal v-model="showLora" />
    <TextualInversionModal v-model="showTI" />
    <StyleEditorModal v-model="showStyle" @saved="onStyleSaved" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { useTxt2VidStore } from '../stores/video'
import { useQuicksettingsStore } from '../stores/quicksettings'
import ResultViewer from '../components/ResultViewer.vue'
import GenerationSettingsCard from '../components/GenerationSettingsCard.vue'
import VideoSettingsCard from '../components/VideoSettingsCard.vue'
import ParamBlocksRenderer from '../components/ParamBlocksRenderer.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import StyleEditorModal from '../components/modals/StyleEditorModal.vue'
import { usePresetsStore } from '../stores/presets'
import { useStylesStore } from '../stores/styles'
import { useUiBlocksStore } from '../stores/ui_blocks'
import CheckpointModal from '../components/modals/CheckpointModal.vue'
import LoraModal from '../components/modals/LoraModal.vue'
import TextualInversionModal from '../components/modals/TextualInversionModal.vue'

const store = useTxt2VidStore()
const presetsStore = usePresetsStore()
const stylesStore = useStylesStore()
const quicksettings = useQuicksettingsStore()
const uiBlocks = useUiBlocksStore()
const leftStack = ref<HTMLElement | null>(null)
const presetName = ref('')
const showStyle = ref(false)
const styleName = ref('')
const showCkpt = ref(false)
const showLora = ref(false)
const showTI = ref(false)
const isWan22 = computed(() => quicksettings.currentEngine === 'wan22')
const isWan22Active = computed(() => uiBlocks.semanticEngine === 'wan22')

onMounted(() => { void Promise.all([store.init(), uiBlocks.init('txt2vid')]) })

onBeforeUnmount(() => {
  store.stopStream()
})

function onSamplerChange(event: Event): void {
  store.sampler = (event.target as HTMLSelectElement).value
}

function onSchedulerChange(event: Event): void {
  store.scheduler = (event.target as HTMLSelectElement).value
}

function onModelChange(event: Event): void {
  void store.setModel((event.target as HTMLSelectElement).value)
}

function onGenerate(event: Event): void {
  event.preventDefault()
  void store.generate()
}

function reset(): void {
  store.prompt = ''
  store.negativePrompt = ''
  store.framesResult = []
  store.info = null
  store.stopStream()
  store.status = 'idle'
  store.errorMessage = ''
  store.progress.stage = 'idle'
  store.progress.percent = null
}

// no aspect ratio selectors; presets and swap available via toolbar

function asJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch (error) {
    return String(value)
  }
}

// presets & styles
const presetNames = computed(() => presetsStore.names('txt2vid'))
function snapshotParams(): Record<string, unknown> {
  return {
    prompt: store.prompt,
    negative: store.negativePrompt,
    steps: store.steps,
    width: store.width,
    height: store.height,
    cfgScale: store.cfgScale,
    seed: store.seed,
    sampler: store.sampler,
    scheduler: store.scheduler,
    frames: store.frames,
    fps: store.fps,
    filenamePrefix: store.filenamePrefix,
    videoFormat: store.videoFormat,
    videoPixFormat: store.videoPixFormat,
    videoCrf: store.videoCrf,
    videoLoopCount: store.videoLoopCount,
    videoPingpong: store.videoPingpong,
    videoSaveMetadata: store.videoSaveMetadata,
    videoSaveOutput: store.videoSaveOutput,
    videoTrimToAudio: store.videoTrimToAudio,
    rifeEnabled: store.rifeEnabled,
    rifeModel: store.rifeModel,
    rifeTimes: store.rifeTimes,
  }
}
function applyParams(v: Record<string, unknown>): void {
  if ('prompt' in v) store.prompt = String(v.prompt)
  if ('negative' in v) store.negativePrompt = String(v.negative)
  if ('steps' in v) store.steps = Number(v.steps)
  if ('width' in v) store.width = Number(v.width)
  if ('height' in v) store.height = Number(v.height)
  if ('cfgScale' in v) store.cfgScale = Number(v.cfgScale)
  if ('seed' in v) store.seed = Number(v.seed)
  if ('sampler' in v) store.sampler = String(v.sampler)
  if ('scheduler' in v) store.scheduler = String(v.scheduler)
  if ('frames' in v) store.frames = Number(v.frames)
  if ('fps' in v) store.fps = Number(v.fps)
  if ('filenamePrefix' in v) store.filenamePrefix = String(v.filenamePrefix)
  if ('videoFormat' in v) store.videoFormat = String(v.videoFormat)
  if ('videoPixFormat' in v) store.videoPixFormat = String(v.videoPixFormat)
  if ('videoCrf' in v) store.videoCrf = Number(v.videoCrf)
  if ('videoLoopCount' in v) store.videoLoopCount = Number(v.videoLoopCount)
  if ('videoPingpong' in v) store.videoPingpong = Boolean(v.videoPingpong)
  if ('videoSaveMetadata' in v) store.videoSaveMetadata = Boolean(v.videoSaveMetadata)
  if ('videoSaveOutput' in v) store.videoSaveOutput = Boolean(v.videoSaveOutput)
  if ('videoTrimToAudio' in v) store.videoTrimToAudio = Boolean(v.videoTrimToAudio)
  if ('rifeEnabled' in v) store.rifeEnabled = Boolean(v.rifeEnabled)
  if ('rifeModel' in v) store.rifeModel = String(v.rifeModel)
  if ('rifeTimes' in v) store.rifeTimes = Number(v.rifeTimes)
}
function savePreset(name: string): void { presetsStore.upsert('txt2vid', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presetsStore.get('txt2vid', name); if (v) applyParams(v) }
const styleNames = computed(() => stylesStore.names())
function applyStyle(name: string): void { const d = stylesStore.get(name); if (!d) return; if (d.prompt) store.prompt += (store.prompt? ' ' : '') + d.prompt; if (d.negative) store.negativePrompt += (store.negativePrompt? ' ' : '') + d.negative }
function onStyleSaved(name: string): void {}
</script>
<style>
/* no view-scoped styles; follows shared components */
</style>
