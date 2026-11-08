<script setup lang="ts">
import type { UIMessage } from 'ai'
import { useClipboard } from '@vueuse/core'
import { getTextFromMessage } from '@nuxt/ui/utils/ai'

const props = defineProps<{ message: UIMessage, streaming: boolean, editing?: boolean, pinned?: boolean }>()
const emit = defineEmits<{ pin: [message: UIMessage] }>()
const clipboard = useClipboard()
const copied = ref(false)
async function copy() {
  await clipboard.copy(getTextFromMessage(props.message))
  copied.value = true
  setTimeout(() => copied.value = false, 2000)
}
</script>

<template>
  <template v-if="message.role === 'assistant' && !streaming">
    <UTooltip text="Copy response">
      <UButton
        size="sm"
        :color="copied ? 'primary' : 'neutral'"
        variant="ghost"
        :icon="copied ? 'i-lucide-copy-check' : 'i-lucide-copy'"
        aria-label="Copy response"
        @click="copy"
      />
    </UTooltip>
    <UTooltip :text="pinned ? 'Unpin answer' : 'Pin answer'">
      <UButton
        size="sm"
        :color="pinned ? 'primary' : 'neutral'"
        variant="ghost"
        :icon="pinned ? 'i-lucide-pin-off' : 'i-lucide-pin'"
        :aria-label="pinned ? 'Unpin answer' : 'Pin answer'"
        @click="emit('pin', message)"
      />
    </UTooltip>
  </template>
</template>
