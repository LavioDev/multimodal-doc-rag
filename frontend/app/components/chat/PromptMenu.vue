<script setup lang="ts">
import type { DropdownMenuItem } from '@nuxt/ui'

const props = defineProps<{
  open: () => void
}>()

defineShortcuts({
  meta_u: {
    usingInput: true,
    handler: () => {
      props.open()
    }
  }
})

const items = computed<DropdownMenuItem[][]>(() => [[{
  label: 'Add files or photos',
  icon: 'i-lucide-paperclip',
  kbds: ['meta', 'U'],
  onSelect: () => props.open()
}]])
</script>

<template>
  <UDropdownMenu
    :items="items"
    :content="{ side: 'top', align: 'start', alignOffset: -4 }"
    :ui="{ content: 'w-52' }"
    size="sm"
  >
    <UButton
      icon="i-lucide-plus"
      color="neutral"
      variant="ghost"
      size="sm"
      aria-label="Open prompt menu"
      class="data-[state=open]:bg-elevated"
    />
  </UDropdownMenu>
</template>
