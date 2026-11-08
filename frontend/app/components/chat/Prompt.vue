<script setup lang="ts">
import type { ChatStatus } from 'ai'

withDefaults(defineProps<{
  status?: ChatStatus
  error?: Error
  files?: FileWithStatus[]
  uploading?: boolean
  disabled?: boolean
  /** Opens the file picker. */
  open: () => void
}>(), {
  status: 'ready',
  error: undefined,
  files: () => []
})

const emit = defineEmits<{
  submit: [event: Event]
  stop: []
  reload: []
  remove: [id: string]
}>()

const input = defineModel<string>({ default: '' })
const mode = defineModel<'text' | 'hybrid'>('mode', { default: 'hybrid' })
const modeItems = computed(() => [[
  { label: 'Hybrid', icon: 'i-lucide-blend', type: 'checkbox' as const, checked: mode.value === 'hybrid', onUpdateChecked: () => mode.value = 'hybrid' },
  { label: 'Text only', icon: 'i-lucide-text', type: 'checkbox' as const, checked: mode.value === 'text', onUpdateChecked: () => mode.value = 'text' }
]])
</script>

<template>
  <UChatPrompt
    v-model="input"
    :status="status"
    :error="error"
    :disabled="uploading || disabled"
    placeholder="Ask about this document..."
    color="neutral"
    variant="subtle"
    :ui="{ base: 'px-1.5' }"
    @submit="emit('submit', $event)"
  >
    <template v-if="files.length > 0" #header>
      <ChatFiles :files="files" @remove="emit('remove', $event)" />
    </template>

    <template #footer>
      <ChatPromptMenu :open="open" />

      <div class="flex items-center gap-1">
        <UDropdownMenu :items="modeItems" :content="{ side: 'top', align: 'end' }">
          <UButton
            color="neutral"
            variant="ghost"
            size="sm"
            trailing-icon="i-lucide-chevron-down"
            :label="mode === 'text' ? 'Text only' : 'Hybrid'"
          />
        </UDropdownMenu>
        <UChatPromptSubmit
          :status="status"
          :disabled="uploading || disabled"
          color="neutral"
          size="sm"
          @stop="emit('stop')"
          @reload="emit('reload')"
        />
      </div>
    </template>
  </UChatPrompt>
</template>
