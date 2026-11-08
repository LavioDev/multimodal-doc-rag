<script setup lang="ts">
const { dropzoneRef, dragging, open, uploading } = useFileUploadWithStatus(async (doc) => {
  await navigateTo(`/chat/${doc.id}`)
})
</script>

<template>
  <UDashboardPanel id="home" class="min-h-0" :ui="{ body: 'p-0 sm:p-0' }">
    <template #header>
      <Navbar />
    </template>
    <template #body>
      <div ref="dropzoneRef" class="flex flex-1">
        <DragDropOverlay :show="dragging" />
        <UContainer class="flex-1 flex items-center justify-center py-8">
          <div
            class="w-full max-w-3xl rounded-2xl border-2 border-dashed border-muted/40 bg-elevated/30 p-8 text-center transition-colors sm:p-12"
            :class="{ 'border-primary bg-primary/5': dragging }"
          >
            <div class="mx-auto flex size-14 items-center justify-center rounded-full bg-primary/10 text-primary">
              <UIcon name="i-lucide-cloud-upload" class="size-7" />
            </div>
            <h1 class="mt-5 text-xl font-semibold text-highlighted sm:text-2xl">
              Add a document to start a chat
            </h1>
            <p class="mt-2 text-sm text-muted">
              Drop a file here or choose one from your device
            </p>
            <UButton
              class="mt-6"
              color="primary"
              icon="i-lucide-file-plus-2"
              :label="uploading ? 'Uploading…' : 'Choose document'"
              :loading="uploading"
              :disabled="uploading"
              @click.stop="open"
            />
            <p class="mt-4 text-xs text-muted">
              PDF, PPTX, or XLSX · Up to 50 MB
            </p>
          </div>
        </UContainer>
      </div>
    </template>
  </UDashboardPanel>
</template>
