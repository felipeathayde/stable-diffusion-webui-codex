<template>
  <section class="panels">
    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Welcome</h2>
      </div>
      <div class="panel-body">
        <p class="muted">
          This home workspace is engine-agnostic. Use it to create and manage model tabs (SD 1.5, SDXL, FLUX, WAN 2.2)
          and to navigate to workflows or utilities. Generation happens in tabs and workflows, not here.
        </p>
        <ul class="list" style="margin-top: .75rem">
          <li class="list-row">
            <div class="list-col grow">
              <strong>Model Tabs</strong>
              <p class="muted">
                Create one or more tabs per engine (e.g., several WAN 2.2 tabs for different model dirs) and open them under
                <code>/models/:tabId</code>. Tabs persist their own parameters.
              </p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>Workflows</strong>
              <p class="muted">
                Use the Workflows view to inspect or run saved workflows built from tab snapshots.
              </p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>SDXL Workspace</strong>
              <p class="muted">
                The SDXL view remains available as a single canonical image workspace for quick runs, separate from tabs.
              </p>
            </div>
          </li>
        </ul>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Create Model Tab</h2>
      </div>
      <div class="panel-body">
        <p class="muted">
          Choose an engine type and an optional title. Tabs are identified by a generated id and can be duplicated or removed later.
        </p>
        <form class="grid grid-3" style="margin-top:.75rem" @submit.prevent="onCreate">
          <div>
            <label class="label" for="engineType">Engine</label>
            <select id="engineType" class="select-md" v-model="newType">
              <option value="sd15">SD 1.5</option>
              <option value="sdxl">SDXL</option>
              <option value="flux">FLUX</option>
              <option value="wan">WAN 2.2</option>
            </select>
          </div>
          <div>
            <label class="label" for="tabTitle">Title (optional)</label>
            <input id="tabTitle" class="ui-input" type="text" v-model="newTitle" placeholder="e.g. WAN — main video rig" />
          </div>
          <div class="grid grid-1" style="align-items:flex-end">
            <button class="btn btn-primary" type="submit">Create Tab</button>
          </div>
        </form>

        <div class="panel-sub" style="margin-top:1rem">
          <h3 class="h5">Existing Tabs</h3>
          <div v-if="!tabs.length" class="muted">No tabs yet. Create one above.</div>
          <ul v-else class="list" style="margin-top:.5rem">
            <li v-for="t in tabs" :key="t.id" class="list-row">
              <div class="list-col grow">
                <RouterLink class="link" :to="`/models/${t.id}`">{{ t.title }}</RouterLink>
                <span class="muted" style="margin-left:.5rem">{{ t.type.toUpperCase() }}</span>
              </div>
              <div class="list-col">
                <button class="btn btn-sm" type="button" @click="dup(t.id)">Duplicate</button>
                <button class="btn btn-sm btn-destructive" type="button" style="margin-left:.5rem" @click="remove(t.id)">Remove</button>
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Docs &amp; Help</h2>
      </div>
      <div class="panel-body">
        <p class="muted">
          The paths below refer to files in this repository and act as the canonical documentation for the Codex UI.
          Open them in your editor when you need deeper guidance. A short Markdown help snippet is also loaded from
          <code>apps/interface/public/help/home-overview.md</code>.
        </p>
        <h3 class="h5" style="margin-top:.75rem">Design &amp; Flows</h3>
        <ul class="list" style="margin-top:.25rem">
          <li class="list-row">
            <div class="list-col grow">
              <strong>Model tabs &amp; workflows spec</strong>
              <p class="muted"><code>.sangoi/design/flows/model-workflows.md</code></p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>Frontend architecture guide</strong>
              <p class="muted"><code>.sangoi/frontend/guidelines/frontend-architecture-guide.md</code></p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>Frontend style guide</strong>
              <p class="muted"><code>.sangoi/frontend/guidelines/frontend-style-guide.md</code></p>
            </div>
          </li>
        </ul>

        <h3 class="h5" style="margin-top:.75rem">Tabs &amp; Workflows Tasks</h3>
        <ul class="list" style="margin-top:.25rem">
          <li class="list-row">
            <div class="list-col grow">
              <strong>F1–F6 tasks</strong>
              <p class="muted">
                <code>.sangoi/tasks/F1-model-tabs-infra.md</code>,
                <code>F2-wan-tab.md</code>,
                <code>F3-image-tabs.md</code>,
                <code>F4-tabs-workflows-backend.md</code>,
                <code>F5-workflows-tab.md</code>,
                <code>F6-refinements.md</code>
              </p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>Open tasks index</strong>
              <p class="muted"><code>.sangoi/tasks/open-tasks-index.md</code></p>
            </div>
          </li>
        </ul>

        <h3 class="h5" style="margin-top:.75rem">Operational References</h3>
        <ul class="list" style="margin-top:.25rem">
          <li class="list-row">
            <div class="list-col grow">
              <strong>Repository structure</strong>
              <p class="muted"><code>.sangoi/architecture/repo-structure.md</code></p>
            </div>
          </li>
          <li class="list-row">
            <div class="list-col grow">
              <strong>Frontend task logs</strong>
              <p class="muted"><code>.sangoi/task-logs/2025-12-03-frontend-*.md</code>, <code>.sangoi/task-logs/2025-12-04-frontend-*.md</code></p>
            </div>
          </li>
        </ul>

        <h3 class="h5" style="margin-top:.75rem">Inline help (Markdown)</h3>
        <div class="toolbar">
          <button
            class="btn btn-sm"
            :class="helpTopic === 'home' ? 'btn-secondary' : 'btn-ghost'"
            type="button"
            @click="setHelpTopic('home')"
          >
            Home
          </button>
          <button
            class="btn btn-sm"
            :class="helpTopic === 'wan22' ? 'btn-secondary' : 'btn-ghost'"
            type="button"
            @click="setHelpTopic('wan22')"
          >
            WAN22 video
          </button>
          <button
            class="btn btn-sm"
            :class="helpTopic === 'workflows' ? 'btn-secondary' : 'btn-ghost'"
            type="button"
            @click="setHelpTopic('workflows')"
          >
            Workflows
          </button>
        </div>
        <MarkdownHelp :src="helpSrc" />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
import MarkdownHelp from '../components/MarkdownHelp.vue'

type HelpTopic = 'home' | 'wan22' | 'workflows'

const router = useRouter()
const store = useModelTabsStore()

const newType = ref<BaseTabType>('sdxl')
const newTitle = ref('')
const helpTopic = ref<HelpTopic>('home')

onMounted(async () => {
  await store.load()
})

const tabs = computed(() => store.orderedTabs)
const helpSrc = computed(() => {
  if (helpTopic.value === 'wan22') return '/help/wan22-quickstart.md'
  if (helpTopic.value === 'workflows') return '/help/workflows-basics.md'
  return '/help/home-overview.md'
})

async function onCreate(): Promise<void> {
  const id = await store.create(newType.value, newTitle.value.trim() || undefined)
  newTitle.value = ''
  if (id) void router.push(`/models/${id}`)
}

function dup(id: string): void {
  void store.duplicate(id)
}

function remove(id: string): void {
  void store.remove(id)
}

function setHelpTopic(topic: HelpTopic): void {
  helpTopic.value = topic
}
</script>
