import type { RagDocument } from './useRagWorkspace'

function uploadError(error: unknown): string {
  const value = error as { data?: { detail?: string }, message?: string }
  return value?.data?.detail || value?.message || 'Upload failed.'
}

export function useFileUploadWithStatus(onUploaded: (document: RagDocument) => void | Promise<void>) {
  const files = ref<FileWithStatus[]>([])
  const toast = useToast()
  const workspace = useRagWorkspace()
  async function uploadFiles(selected: File[]) {
    const file = selected[0]
    if (!file) return
    if (!/\.(pdf|pptx|xlsx)$/i.test(file.name)) {
      toast.add({ title: 'Unsupported file', description: 'Choose a PDF, PPTX, or XLSX file.', color: 'error' })
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.add({ title: 'File is too large', description: 'Maximum file size is 50 MB.', color: 'error' })
      return
    }
    files.value.forEach(item => URL.revokeObjectURL(item.previewUrl))
    const item: FileWithStatus = { file, id: crypto.randomUUID(), previewUrl: URL.createObjectURL(file), status: 'uploading' }
    files.value = [item]
    try {
      const document = await workspace.upload(file)
      files.value = [{ ...item, status: 'uploaded' }]
      await onUploaded(document)
    } catch (error: unknown) {
      const message = uploadError(error)
      files.value = [{ ...item, status: 'error', error: message }]
      toast.add({ title: 'Upload failed', description: message, color: 'error', icon: 'i-lucide-alert-circle' })
    }
  }
  const { dropzoneRef, isDragging, open } = useFileUpload({ accept: FILE_UPLOAD_CONFIG.acceptPattern, multiple: false, onUpdate: uploadFiles })
  const uploading = computed(() => files.value.some(file => file.status === 'uploading'))
  function removeFile(id: string) {
    const file = files.value.find(item => item.id === id)
    if (file) URL.revokeObjectURL(file.previewUrl)
    files.value = files.value.filter(item => item.id !== id)
  }
  function clearFiles() {
    files.value.forEach(file => URL.revokeObjectURL(file.previewUrl))
    files.value = []
  }
  onUnmounted(clearFiles)
  return { dropzoneRef, dragging: isDragging, open, files, uploading, removeFile, clearFiles, addFiles: uploadFiles }
}
