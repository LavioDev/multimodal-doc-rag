<script setup lang="ts">
import type { UIMessage } from 'ai'

const route = useRoute()
const workspace = useRagWorkspace()
const toast = useToast()
const docId = computed(() => String(route.params.id))
const { dropzoneRef, dragging, open, files, uploading, removeFile } = useFileUploadWithStatus(async (nextDocument) => {
  await navigateTo(`/chat/${nextDocument.id}`)
})
const { data: document, error: documentRequestError, refresh: refreshDocument } = await useFetch<RagDocument>(() => `/api/rag/documents/${docId.value}`, { key: `rag-document-${docId.value}`, default: () => undefined })
const turns = workspace.turns(docId.value)
const persistedMessages = workspace.messages(docId.value)
const messages = computed<UIMessage[]>(() => {
  const pending = workspace.state.value.pending[docId.value]
  if (!pending || !workspace.state.value.busy[docId.value]) return persistedMessages.value
  return [...persistedMessages.value, { id: `${pending.request_id}-pending`, role: 'user', parts: [{ type: 'text', text: pending.question }] } as UIMessage]
})
const input = computed({ get: () => workspace.state.value.drafts[docId.value] || '', set: value => workspace.state.value.drafts[docId.value] = value })
const mode = computed({ get: () => workspace.state.value.modes[docId.value] || 'hybrid', set: value => workspace.state.value.modes[docId.value] = value })
const loading = computed(() => workspace.state.value.busy[docId.value] === true)
const error = computed(() => workspace.state.value.errors[docId.value])
const loaded = computed(() => workspace.state.value.loaded[docId.value] === true)
const showStarter = computed(() => loaded.value && messages.value.length === 0)
const ready = computed(() => document.value?.status === 'ready')
const processable = computed(() => !!document.value && ['uploaded', 'prepared', 'failed'].includes(document.value.status))
const processing = computed(() => ['preparing', 'queued', 'processing'].includes(document.value?.status || ''))
const progress = computed(() => document.value?.page_count ? Math.min(100, Math.round((document.value.progress_pages / document.value.page_count) * 100)) : 0)
const statusLabel = computed(() => document.value?.status === 'ready' ? 'Ready' : document.value?.status === 'processing' ? `Processing · ${document.value.progress_pages}/${document.value.page_count || '?'} pages` : document.value?.status || 'Loading')
const pins = computed(() => workspace.state.value.pins[docId.value] || [])
const pinsOpen = useState('rag-pins-open', () => false)
const pinnedTurnIds = computed(() => new Set(pins.value.map(pin => pin.turn_id)))
const turnByMessage = computed(() => new Map(turns.value.map(turn => [turn.id, turn])))
const sourceOpen = ref(false)
const sourcePage = ref<number | null>(null)
const sourceLoading = ref(false)
const sourceError = ref<string | null>(null)
const sourceEvidence = ref<RagEvidence[]>([])
const sourceCitation = ref<RagCitation | null>(null)
const pageImageOpen = ref(false)
const sourceImageAspect = ref(1)
const pageImageSize = ref({ width: 0, height: 0 })
const pageViewerSize = ref({ width: 1280, height: 800 })
const pageZoom = ref(1)
const suggestedQuestions = [
  'What is this document about?',
  'Summarize the key findings',
  'What evidence supports the main conclusion?',
  'List the important dates and figures',
  'Compare the main points',
  'Explain the charts and tables',
  'What should I look at first?'
]
const imageUrl = computed(() => sourcePage.value ? `/api/rag/documents/${docId.value}/pages/${sourcePage.value}/image` : '')
const totalPages = computed(() => document.value?.page_count || 1)
const pageDisplayWidth = computed(() => {
  const { width, height } = pageImageSize.value
  if (!width || !height) return Math.min(pageViewerSize.value.width - 48, 900) * pageZoom.value
  const maxWidth = Math.max(280, pageViewerSize.value.width - 48)
  const maxHeight = Math.max(280, pageViewerSize.value.height - 128)
  return Math.min(maxWidth, maxHeight * width / height) * pageZoom.value
})
const sourceBbox = computed(() => sourceCitation.value?.bbox || sourceEvidence.value.find(item => item.bbox)?.bbox || null)
const normalizedSourceBbox = computed(() => {
  if (!sourceBbox.value || sourceBbox.value.length !== 4) return null
  const values = sourceBbox.value.map(value => value > 1 ? value / 100 : value)
  const [x1, y1, x2, y2] = values
  if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return null
  const left = Math.max(0, Math.min(1, x1))
  const top = Math.max(0, Math.min(1, y1))
  const right = Math.max(left, Math.min(1, x2))
  const bottom = Math.max(top, Math.min(1, y2))
  if (right - left < 0.005 || bottom - top < 0.005) return null
  return { left, top, width: right - left, height: bottom - top }
})
const sourceCropBox = computed(() => {
  const box = normalizedSourceBbox.value
  if (!box) return null
  const padX = Math.max(box.width * 0.08, 0.015)
  const padY = Math.max(box.height * 0.1, 0.02)
  const left = Math.max(0, box.left - padX)
  const top = Math.max(0, box.top - padY)
  const right = Math.min(1, box.left + box.width + padX)
  const bottom = Math.min(1, box.top + box.height + padY)
  return { left, top, width: right - left, height: bottom - top }
})
const sourceCropStyle = computed(() => {
  const box = sourceCropBox.value
  if (!box) return {}
  return {
    left: `${-box.left / box.width * 100}%`,
    top: `${-box.top / box.height * 100}%`,
    width: `${100 / box.width}%`,
    height: `${100 / box.height}%`
  }
})
const sourcePreviewStyle = computed(() => {
  const box = sourceCropBox.value
  return box ? { aspectRatio: `${box.width * sourceImageAspect.value} / ${box.height}` } : undefined
})
function updateSourceImageAspect(event: Event) {
  const image = event.target as HTMLImageElement
  if (image.naturalWidth && image.naturalHeight) sourceImageAspect.value = image.naturalWidth / image.naturalHeight
}
function updatePageImageSize(event: Event) {
  const image = event.target as HTMLImageElement
  if (image.naturalWidth && image.naturalHeight) pageImageSize.value = { width: image.naturalWidth, height: image.naturalHeight }
}
function updatePageViewerSize() {
  pageViewerSize.value = { width: window.innerWidth, height: window.innerHeight }
}
const sourceBoxStyle = computed(() => {
  const box = normalizedSourceBbox.value
  if (!box) return {}
  return { left: `${box.left * 100}%`, top: `${box.top * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }
})
function routeErrorMessage(error: unknown, fallback: string): string {
  const value = error as { data?: { detail?: string }, message?: string }
  return value?.data?.detail || value?.message || fallback
}
async function openSource(page: number, evidence: RagEvidence[] = [], citation?: RagCitation) {
  pageImageOpen.value = false
  sourcePage.value = page
  sourceEvidence.value = evidence
  sourceCitation.value = citation || null
  sourceError.value = null
  sourceOpen.value = true
  sourceLoading.value = true
  try {
    const pageData = await workspace.call<Array<{ page: number }>>(`documents/${docId.value}/pages`)
    const data = pageData.find(item => item.page === page)
    if (!data) sourceError.value = 'Source page was not found.'
  } catch (error: unknown) {
    sourceError.value = routeErrorMessage(error, 'Could not load source page.')
  } finally {
    sourceLoading.value = false
  }
}
function closePageImage() {
  pageImageOpen.value = false
}
function openFullPage() {
  pageZoom.value = 1
  pageImageOpen.value = true
}
function openDocument() {
  sourcePage.value = 1
  sourceEvidence.value = []
  sourceCitation.value = null
  pageImageSize.value = { width: 0, height: 0 }
  pageZoom.value = 1
  pageImageOpen.value = true
}
function changeSourcePage(offset: number) {
  if (sourcePage.value === null) return
  sourcePage.value = Math.max(1, Math.min(totalPages.value, sourcePage.value + offset))
  sourceEvidence.value = []
  sourceCitation.value = null
  pageImageSize.value = { width: 0, height: 0 }
}
function zoomPage(offset: number) {
  pageZoom.value = Math.max(0.5, Math.min(3, Math.round((pageZoom.value + offset) * 100) / 100))
}
defineShortcuts({ escape: closePageImage })
async function submit() {
  if (!ready.value || !loaded.value || !input.value.trim()) return
  await workspace.send(docId.value)
}
function useSuggestion(question: string) {
  input.value = question
}
async function process() {
  try {
    await workspace.processDocument(docId.value)
    await refreshDocument()
  } catch (error: unknown) {
    toast.add({ color: 'error', icon: 'i-lucide-alert-circle', description: routeErrorMessage(error, 'Could not process document.') })
  }
}
async function togglePin(message: UIMessage) {
  const turn = turnByMessage.value.get(message.id)
  if (!turn) return
  const existing = pins.value.find(pin => pin.turn_id === turn.id)
  try {
    if (existing) await workspace.unpin(docId.value, existing.id)
    else await workspace.pin(docId.value, turn.id)
  } catch (error: unknown) {
    toast.add({ color: 'error', icon: 'i-lucide-alert-circle', description: routeErrorMessage(error, 'Could not update pin.') })
  }
}
onMounted(async () => {
  updatePageViewerSize()
  window.addEventListener('resize', updatePageViewerSize)
  await Promise.all([workspace.refreshDocuments(), workspace.loadDocument(docId.value)])
  if (!document.value) await refreshDocument()
  const pending = workspace.state.value.pending[docId.value]
  if (pending && workspace.state.value.drafts[docId.value] === undefined) workspace.state.value.drafts[docId.value] = pending.question
})
onUnmounted(() => window.removeEventListener('resize', updatePageViewerSize))
watch(docId, async (id) => {
  sourceOpen.value = false
  await workspace.loadDocument(id)
  await refreshDocument()
})
watch(() => workspace.state.value.pagePreview, (preview) => {
  if (preview?.documentId === docId.value) {
    openSource(preview.page, preview.evidence, preview.citation)
    workspace.state.value.pagePreview = null
  }
})
watch(() => document.value?.status, (status) => {
  if (status && ['queued', 'processing', 'preparing'].includes(status)) {
    const timer = window.setInterval(async () => {
      await refreshDocument()
      await workspace.refreshDocuments()
    }, 1800)
    const stop = watch(() => document.value?.status, (next) => {
      if (!next || !['queued', 'processing', 'preparing'].includes(next)) {
        clearInterval(timer)
        stop()
      }
    })
  }
})
</script>

<template>
  <div class="contents">
    <UDashboardPanel
      v-if="document"
      id="chat"
      class="relative min-h-0"
      :ui="{ body: 'p-0 sm:p-0 overscroll-none' }"
    >
      <template #header>
        <Navbar>
          <template #title>
            <div class="flex min-w-0 items-center gap-2">
              <span class="truncate text-sm font-medium text-highlighted">{{ document.filename }}</span><UBadge
                size="xs"
                color="neutral"
                variant="subtle"
                class="capitalize"
              >
                {{ statusLabel }}
              </UBadge>
            </div>
          </template>
          <UTooltip v-if="ready" text="View document">
            <UButton
              icon="i-lucide-book-open"
              color="neutral"
              variant="ghost"
              aria-label="View document"
              @click="openDocument"
            />
          </UTooltip>
          <UButton
            v-if="ready"
            icon="i-lucide-bookmark"
            color="neutral"
            variant="ghost"
            :label="pins.length ? `${pins.length}` : undefined"
            aria-label="Pinned answers"
            @click="pinsOpen = true"
          />
        </Navbar>
      </template>
      <template #body>
        <div ref="dropzoneRef" class="flex min-h-0 min-w-0 flex-1" :class="sourceOpen ? 'lg:pr-[min(520px,45vw)]' : ''">
          <DragDropOverlay :show="dragging" />
          <UContainer class="min-w-0 flex-1 flex flex-col gap-4 sm:gap-6">
            <div v-if="processable || processing || !ready" class="pt-(--ui-header-height) space-y-3">
              <div class="flex items-center justify-between gap-4">
                <div class="min-w-0">
                  <p class="font-medium">
                    {{ document.filename }}
                  </p><p class="text-sm text-muted">
                    {{ statusLabel }}<span v-if="document.page_count"> · {{ document.page_count }} pages</span>
                  </p>
                </div><UButton
                  v-if="processable"
                  :label="document.status === 'failed' ? 'Retry' : 'Process'"
                  icon="i-lucide-play"
                  color="primary"
                  :loading="loading"
                  @click="process"
                />
              </div>
              <UProgress v-if="processing" :model-value="progress" />
              <UAlert
                v-if="document.error"
                color="error"
                variant="subtle"
                icon="i-lucide-alert-circle"
                :description="document.error"
              />
            </div>
            <UChatMessages
              v-if="!showStarter"
              should-auto-scroll
              :messages="messages"
              :status="loading ? 'submitted' : 'ready'"
              :spacing-offset="200"
              class="pt-(--ui-header-height) pb-4 sm:pb-6"
            >
              <template #indicator>
                <div
                  class="flex items-center gap-1.5"
                  role="status"
                  aria-live="polite"
                  aria-label="Waiting for an answer"
                >
                  <ChatIndicator /><UChatShimmer text="Thinking..." class="text-sm" />
                </div>
              </template>
              <template #content="{ message }">
                <div :id="message.role === 'assistant' ? `turn-${turnByMessage.get(message.id)?.id}` : undefined">
                  <ChatMessageContent
                    :message="message"
                    :sources="turnByMessage.get(message.id)?.source_pages"
                    :evidence="turnByMessage.get(message.id)?.evidence"
                    :answer-parts="turnByMessage.get(message.id)?.answer_parts"
                    @source="openSource"
                  />
                </div>
              </template>
              <template #actions="{ message }">
                <ChatMessageActions
                  :message="message"
                  :streaming="loading && message.id === messages[messages.length - 1]?.id"
                  :pinned="pinnedTurnIds.has(turnByMessage.get(message.id)?.id || '')"
                  @pin="togglePin"
                />
              </template>
            </UChatMessages>
            <template v-if="showStarter">
              <div class="flex flex-1 flex-col items-center justify-center py-8">
                <div class="w-full max-w-3xl space-y-3">
                  <ChatPrompt
                    v-model="input"
                    v-model:mode="mode"
                    :status="loading ? 'submitted' : 'ready'"
                    :disabled="!ready || !loaded || loading"
                    :files="files"
                    :uploading="uploading"
                    :error="error ? new Error(error) : undefined"
                    :open="open"
                    class="[view-transition-name:chat-prompt]"
                    @submit="submit"
                    @remove="removeFile"
                  />
                  <div class="flex flex-wrap justify-center gap-2">
                    <UButton
                      v-for="question in suggestedQuestions"
                      :key="question"
                      :label="question"
                      size="sm"
                      color="neutral"
                      variant="outline"
                      class="rounded-full"
                      :disabled="!ready || !loaded || loading"
                      @click="useSuggestion(question)"
                    />
                  </div>
                  <p v-if="!ready" class="text-center text-sm text-muted">
                    Process this document to start asking questions.
                  </p>
                </div>
              </div>
            </template>
            <template v-else>
              <div v-if="error" class="flex items-center justify-between gap-2 rounded-md border border-error/30 bg-error/5 px-3 py-2 text-sm">
                <span class="text-error">{{ error }}</span><UButton
                  size="xs"
                  label="Retry"
                  color="error"
                  variant="soft"
                  :loading="loading"
                  @click="loaded ? submit() : workspace.loadDocument(docId)"
                />
              </div>
              <ChatPrompt
                v-model="input"
                v-model:mode="mode"
                :status="loading ? 'submitted' : 'ready'"
                :disabled="!ready || !loaded || loading"
                :files="files"
                :uploading="uploading"
                :error="error ? new Error(error) : undefined"
                :open="open"
                class="sticky bottom-0 [view-transition-name:chat-prompt] rounded-b-none z-10"
                @submit="submit"
                @remove="removeFile"
              />
              <p v-if="!loaded && !workspace.state.value.errors[docId]" class="text-center text-sm text-muted">
                Loading chat history...
              </p>
            </template>
          </UContainer>
          <aside
            v-if="sourceOpen"
            class="fixed inset-0 z-50 flex flex-col border-l border-default bg-default lg:left-auto lg:w-[min(520px,45vw)]"
            aria-label="Source evidence"
          >
            <div class="flex items-start justify-between gap-3 border-b border-default p-4">
              <div class="min-w-0">
                <h2 class="font-semibold">
                  Source · Page {{ sourcePage }}
                </h2>
                <p class="truncate text-xs text-muted">
                  {{ document?.filename }}
                </p>
              </div>
              <UButton
                icon="i-lucide-x"
                variant="ghost"
                color="neutral"
                aria-label="Close source"
                @click="sourceOpen = false"
              />
            </div>
            <div class="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 sm:p-5">
              <p v-if="sourceError" class="text-sm text-error">
                {{ sourceError }}
              </p>
              <div v-if="sourceLoading" class="flex h-48 items-center justify-center">
                <UProgress class="w-40" />
              </div>
              <template v-else>
                <div v-if="sourceCitation?.excerpt" class="rounded-xl border border-default bg-elevated p-4 shadow-sm">
                  <p class="mb-1 text-xs font-semibold text-primary">
                    Source excerpt
                  </p>
                  <p class="whitespace-pre-wrap text-sm">
                    {{ sourceCitation.excerpt }}
                  </p>
                  <p v-if="sourceCitation.source_kind === 'ocr'" class="mt-2 text-xs text-muted">
                    Extracted with OCR
                  </p>
                </div>
                <div v-else-if="sourceCitation?.source_kind === 'visual'" class="rounded-xl border border-default bg-elevated p-4 text-sm text-muted shadow-sm">
                  {{ normalizedSourceBbox ? 'Image evidence · Figure extracted from this page. Click the image to open the full page.' : 'Image evidence · No extractable text for this citation. Inspect the page image below.' }}
                </div>
                <div v-else-if="sourceEvidence.length" class="space-y-2">
                  <p v-for="(item, index) in sourceEvidence" :key="index" class="border-l-2 border-primary pl-2 text-sm">
                    {{ item.text }}
                  </p>
                </div>
                <button
                  v-if="!sourceError"
                  type="button"
                  class="group relative block w-full overflow-hidden rounded-xl border border-default bg-muted/20 text-left shadow-sm transition hover:border-primary/50 hover:shadow-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  :style="sourcePreviewStyle"
                  :aria-label="`Open full page ${sourcePage}`"
                  @click="openFullPage"
                >
                  <img
                    v-if="normalizedSourceBbox"
                    :src="imageUrl"
                    :alt="`Preview of cited content on page ${sourcePage}`"
                    class="absolute max-w-none"
                    :style="sourceCropStyle"
                    @load="updateSourceImageAspect"
                  >
                  <img
                    v-else
                    :src="imageUrl"
                    :alt="`Source page ${sourcePage}`"
                    class="mx-auto block h-auto max-h-[70vh] w-full object-contain"
                  >
                  <span class="absolute bottom-2 right-2 inline-flex items-center gap-1.5 rounded-full bg-default/90 px-3 py-1.5 text-xs font-medium text-highlighted opacity-90 shadow transition group-hover:opacity-100">
                    <UIcon name="i-lucide-maximize-2" class="size-3.5" />
                    Open full page
                  </span>
                </button>
              </template>
            </div>
          </aside>
        </div>
      </template>
    </UDashboardPanel>
    <UContainer v-else-if="documentRequestError || workspace.state.value.errors[docId]?.includes('404')" class="flex-1 flex flex-col gap-4 sm:gap-6">
      <UError :error="{ statusMessage: 'Document not found', statusCode: 404 }" class="min-h-full" />
    </UContainer>
    <UContainer v-else class="flex-1 flex items-center justify-center">
      <UProgress class="w-48" />
    </UContainer>
  </div>
  <Teleport to="body">
    <div
      v-if="pageImageOpen"
      class="fixed inset-0 z-[100] flex flex-col bg-black/90 p-3 text-white backdrop-blur-sm sm:p-5"
      role="dialog"
      aria-modal="true"
      :aria-label="`Full page ${sourcePage}`"
    >
      <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/15 px-1 pb-3">
        <div class="min-w-0">
          <p class="truncate text-sm font-medium">
            {{ document?.filename }}
          </p>
          <p class="text-xs text-white/65">
            Full page view
          </p>
        </div>
        <div class="flex flex-wrap items-center justify-center gap-2">
          <UButton
            icon="i-lucide-chevron-left"
            color="neutral"
            variant="soft"
            aria-label="Previous page"
            :disabled="(sourcePage || 1) <= 1"
            @click="changeSourcePage(-1)"
          />
          <span class="min-w-24 text-center text-sm tabular-nums">Page {{ sourcePage }} / {{ totalPages }}</span>
          <UButton
            icon="i-lucide-chevron-right"
            color="neutral"
            variant="soft"
            aria-label="Next page"
            :disabled="(sourcePage || 1) >= totalPages"
            @click="changeSourcePage(1)"
          />
          <span class="mx-1 h-6 border-l border-white/20" />
          <UButton
            icon="i-lucide-minus"
            color="neutral"
            variant="soft"
            aria-label="Zoom out"
            :disabled="pageZoom <= 0.5"
            @click="zoomPage(-0.25)"
          />
          <span class="w-12 text-center text-sm tabular-nums">{{ Math.round(pageZoom * 100) }}%</span>
          <UButton
            icon="i-lucide-plus"
            color="neutral"
            variant="soft"
            aria-label="Zoom in"
            :disabled="pageZoom >= 3"
            @click="zoomPage(0.25)"
          />
          <UButton
            icon="i-lucide-x"
            color="neutral"
            variant="solid"
            aria-label="Close full page"
            @click="closePageImage"
          />
        </div>
      </div>
      <div class="min-h-0 flex-1 overflow-auto p-2 sm:p-4">
        <div class="flex min-h-full w-fit min-w-full items-center justify-center">
          <div class="relative shrink-0" :style="{ width: `${pageDisplayWidth}px` }">
            <img
              :src="imageUrl"
              :alt="`Full source page ${sourcePage}`"
              class="block h-auto w-full rounded-sm shadow-2xl"
              @load="updatePageImageSize"
            >
            <div
              v-if="normalizedSourceBbox"
              class="pointer-events-none absolute border-2 border-primary bg-transparent shadow-none"
              :style="sourceBoxStyle"
              aria-hidden="true"
            />
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
