export interface FileWithStatus {
  file: File
  id: string
  previewUrl: string
  status: 'uploading' | 'uploaded' | 'error'
  uploadedUrl?: string
  uploadedPathname?: string
  error?: string
}

export const FILE_UPLOAD_CONFIG = {
  maxSize: '50MB',
  types: ['application/pdf', '.pptx', '.xlsx'],
  acceptPattern: '.pdf,.pptx,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
} as const

export function getFileIcon(mimeType: string, fileName?: string): string {
  if (mimeType.startsWith('image/')) return 'i-lucide-image'
  if (mimeType === 'application/pdf') return 'i-lucide-file-text'
  if (mimeType.includes('spreadsheet') || mimeType.includes('presentation') || /\.(xlsx|pptx)$/i.test(fileName || '')) return 'i-lucide-file-spreadsheet'
  return 'i-lucide-file'
}

export function removeRandomSuffix(filename: string): string {
  return filename.replace(/^(.+)-[a-zA-Z0-9]+(\.[^.]+)$/, '$1$2')
}
