<script setup lang="ts">
import type { Ref } from 'vue'
import type { UIChat } from '~/composables/useChats'

const workspace = useRagWorkspace()
const sidebarOpen = ref(false)
const searchOpen = ref(false)
const route = useRoute()
const { data: ragDocs } = await useFetch<RagDocument[]>('/api/rag/documents', { key: 'rag-documents', default: () => [] })
watch(ragDocs, (docs) => {
  workspace.state.value.documents = docs || []
}, { immediate: true })
const chats = computed(() => workspace.documents.value.map(document => ({
  id: document.id,
  label: document.filename,
  to: `/chat/${document.id}`,
  icon: document.status === 'ready' ? 'i-lucide-file-check-2' : document.status === 'failed' ? 'i-lucide-file-warning' : 'i-lucide-file-text',
  createdAt: document.created_at,
  status: document.status,
  progress: document.progress_pages,
  total: document.page_count || 0
})))
const { groups } = useChats(chats as Ref<UIChat[] | undefined>)
const items = computed(() => groups.value.flatMap(group => [{ label: group.label, type: 'label' as const }, ...group.items.map(item => ({
  ...item,
  slot: 'chat' as const,
  icon: item.icon,
  class: item.status === 'ready' ? '' : 'text-muted'
}))]))
const searchGroups = computed(() => [{ id: 'actions', items: [{ label: 'New chat', to: '/', icon: 'i-lucide-circle-plus', kbds: ['meta', 'o'] }] }, ...groups.value])
const pinsOpen = useState('rag-pins-open', () => false)
const currentId = computed(() => String(route.params.id || ''))
const currentPins = computed(() => workspace.state.value.pins[currentId.value] || [])
const pinExpanded = ref<string[]>([])
const deletingPin = ref<string | null>(null)
function jumpToTurn(id: string) {
  pinsOpen.value = false
  nextTick(() => document.getElementById(`turn-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}
async function removePin(pin: RagPin) {
  deletingPin.value = pin.id
  try {
    await workspace.unpin(currentId.value, pin.id)
  } finally {
    deletingPin.value = null
  }
}
defineShortcuts({ meta_o: () => navigateTo('/'), meta_k: () => searchOpen.value = true })
onMounted(() => {
  const timer = window.setInterval(() => {
    if (workspace.documents.value.some(doc => ['queued', 'processing', 'preparing'].includes(doc.status))) {
      workspace.refreshDocuments()
    }
  }, 1800)
  onBeforeUnmount(() => window.clearInterval(timer))
})
</script>

<template>
  <div class="contents">
    <UDashboardGroup unit="rem">
      <UDashboardSidebar
        id="default"
        v-model:open="sidebarOpen"
        :min-size="12"
        collapsible
        resizable
        :menu="{ inset: true }"
        class="border-r-0 py-4 dark:[--ui-bg-elevated:var(--ui-color-neutral-900)]"
      >
        <template #header="{ collapsed }">
          <NuxtLink v-if="!collapsed" to="/" class="flex items-end gap-1 outline-primary/25 focus-visible:outline-3 rounded-md"><span class="text-xl font-bold text-highlighted ml-2">Docs RAG</span></NuxtLink>
          <UDashboardSidebarCollapse class="ms-auto" />
        </template>
        <template #default="{ collapsed }">
          <UNavigationMenu :items="[{ label: 'New chat', to: '/', kbds: ['meta', 'o'], icon: 'i-lucide-circle-plus' }, { label: 'Search', icon: 'i-lucide-search', kbds: ['meta', 'k'], onSelect: () => searchOpen = true }]" :collapsed="collapsed" orientation="vertical">
            <template #item-trailing="{ item }">
              <div v-if="item.kbds?.length" class="flex items-center gap-px opacity-0 group-hover:opacity-100 transition-opacity">
                <UKbd
                  v-for="kbd in item.kbds"
                  :key="kbd"
                  :value="kbd"
                  size="sm"
                  variant="soft"
                  class="bg-accented/50"
                />
              </div>
            </template>
          </UNavigationMenu>
          <UNavigationMenu
            v-if="!collapsed"
            :items="items"
            :collapsed="collapsed"
            orientation="vertical"
            :ui="{ link: 'overflow-hidden pr-7.5', linkTrailing: 'translate-x-full group-hover:translate-x-0 transition-transform ms-0 absolute inset-e-px' }"
          >
            <template #chat-trailing="{ item }">
              <UTooltip :text="`${item.status}${item.status === 'processing' ? ` · ${item.progress}/${item.total} pages` : ''}`">
                <span class="text-[10px] capitalize text-muted">{{ item.status === 'processing' ? `${item.progress}/${item.total}` : item.status }}</span>
              </UTooltip>
            </template>
          </UNavigationMenu>
        </template>
      </UDashboardSidebar>
      <UDashboardSearch v-model:open="searchOpen" placeholder="Search documents..." :groups="searchGroups" />
      <div class="flex-1 flex m-4 lg:ml-0 rounded-lg ring ring-default bg-default/75 shadow-sm min-w-0 overflow-hidden">
        <slot />
      </div>
    </UDashboardGroup>

    <UModal
      v-model:open="pinsOpen"
      title="Pinned answers"
      :description="currentPins.length ? `${currentPins.length} saved for this document` : 'No pinned answers yet.'"
      :ui="{ content: 'sm:max-w-2xl' }"
    >
      <template #content>
        <div class="p-5 space-y-3 max-h-[75vh] overflow-y-auto">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">
              Pinned answers
            </h2><UButton
              icon="i-lucide-x"
              color="neutral"
              variant="ghost"
              aria-label="Close"
              @click="pinsOpen = false"
            />
          </div>
          <div v-if="!currentPins.length" class="py-8 text-center text-muted">
            No pinned answers yet.
          </div>
          <UCard v-for="pin in currentPins" :key="pin.id">
            <div class="flex justify-between gap-3">
              <div class="min-w-0">
                <p class="font-medium">
                  {{ pin.question }}
                </p><div v-if="pin.answer_parts?.length" class="mt-2 space-y-2 text-sm text-muted" :class="!pinExpanded.includes(pin.id) && 'line-clamp-3'">
                  <p v-for="(part, partIndex) in pin.answer_parts" :key="partIndex">
                    {{ part.text }}
                    <button
                      v-for="(citation, citationIndex) in part.citations"
                      :key="citationIndex"
                      type="button"
                      class="ml-1 font-semibold text-primary underline"
                      :aria-label="`Show source page ${citation.page}`"
                      @click="pinsOpen = false; workspace.state.value.pagePreview = { documentId: currentId, page: citation.page, evidence: citation.excerpt ? [{ doc_id: citation.doc_id, page: citation.page, text: citation.excerpt, bbox: citation.bbox }] : [], citation }"
                    >
                      [p.{{ citation.page }}]
                    </button>
                    <span v-if="part.status === 'unverified'" class="ml-1 text-warning">Unverified</span>
                  </p>
                </div><p v-else class="mt-2 text-sm text-muted line-clamp-3" :class="pinExpanded.includes(pin.id) && '!line-clamp-none'">
                  {{ pin.answer }}
                </p><div v-if="!pin.answer_parts?.length" class="mt-2 flex flex-wrap gap-1">
                  <UButton
                    v-for="source in pin.source_pages"
                    :key="source.page"
                    size="xs"
                    variant="outline"
                    color="neutral"
                    :label="`Page ${source.page}`"
                    @click="workspace.state.value.pagePreview = { documentId: currentId, page: source.page, evidence: pin.evidence.filter(e => e.page === source.page) }"
                  />
                </div>
              </div><div class="flex shrink-0 items-start">
                <UButton
                  size="xs"
                  variant="ghost"
                  color="neutral"
                  :icon="pinExpanded.includes(pin.id) ? 'i-lucide-chevron-up' : 'i-lucide-chevron-down'"
                  aria-label="Expand answer"
                  @click="pinExpanded = pinExpanded.includes(pin.id) ? pinExpanded.filter(id => id !== pin.id) : [...pinExpanded, pin.id]"
                /><UButton
                  size="xs"
                  variant="ghost"
                  color="neutral"
                  icon="i-lucide-message-square"
                  aria-label="Go to original answer"
                  @click="jumpToTurn(pin.turn_id)"
                /><UButton
                  size="xs"
                  variant="ghost"
                  color="error"
                  icon="i-lucide-pin-off"
                  aria-label="Unpin answer"
                  :loading="deletingPin === pin.id"
                  @click="removePin(pin)"
                />
              </div>
            </div>
          </UCard>
        </div>
      </template>
    </UModal>
  </div>
</template>
