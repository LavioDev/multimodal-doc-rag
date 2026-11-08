<script setup lang="ts">
import type { UIMessage } from 'ai'
import type { RagAnswerPart, RagCitation, RagEvidence, RagSource } from '~/composables/useRagWorkspace'
import { isTextUIPart } from 'ai'
import { isPartStreaming } from '@nuxt/ui/utils/ai'

defineProps<{ message: UIMessage, editing?: boolean, sources?: RagSource[], evidence?: RagEvidence[], answerParts?: RagAnswerPart[] }>()
const emit = defineEmits<{ source: [page: number, evidence: RagEvidence[], citation?: RagCitation] }>()
</script>

<template>
  <div v-if="message.role === 'assistant' && answerParts?.length" class="space-y-3">
    <p v-for="(answerPart, index) in answerParts" :key="index" class="whitespace-pre-wrap leading-relaxed">
      {{ answerPart.text }}
      <button
        v-for="(citation, citationIndex) in answerPart.citations"
        :key="`${citation.doc_id}-${citation.page}-${citationIndex}`"
        type="button"
        class="ml-1 rounded text-sm font-semibold text-primary underline underline-offset-2 hover:text-primary/80 focus-visible:outline-2 focus-visible:outline-primary"
        :aria-label="`Show source page ${citation.page} for this claim`"
        @click="emit('source', citation.page, citation.excerpt ? [{ doc_id: citation.doc_id, page: citation.page, text: citation.excerpt, bbox: citation.bbox }] : [], citation)"
      >
        [p.{{ citation.page }}]
      </button>
      <span v-if="answerPart.status === 'unverified'" class="ml-1 inline-flex items-center gap-1 rounded bg-warning/10 px-1.5 py-0.5 text-xs font-medium text-warning" title="The cited material has not been confirmed to support this claim.">
        <UIcon name="i-lucide-triangle-alert" class="size-3" />Unverified
      </span>
    </p>
  </div>
  <template v-for="(part, index) in getMergedParts(message.parts)" :key="`${message.id}-${index}`">
    <ChatComark v-if="isTextUIPart(part) && message.role === 'assistant' && !answerParts?.length" :value="part.text" :streaming="isPartStreaming(part)" />
    <p v-else-if="isTextUIPart(part) && message.role === 'user'" class="whitespace-pre-wrap">
      {{ part.text }}
    </p>
  </template>
  <div v-if="message.role === 'assistant' && !answerParts?.length && sources?.length" class="mt-3 flex flex-wrap gap-1.5">
    <UButton
      v-for="source in sources"
      :key="source.page"
      size="xs"
      variant="outline"
      color="neutral"
      icon="i-lucide-book-open"
      :label="`Page ${source.page}`"
      @click="emit('source', source.page, evidence?.filter(item => item.page === source.page) || [])"
    />
  </div>
</template>
