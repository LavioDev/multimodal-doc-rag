import type { UIMessage } from 'ai'

export interface RagDocument {
  id: string
  filename: string
  size_bytes: number
  status: string
  page_count: number | null
  progress_pages: number
  error: string | null
  created_at: string
}

export interface RagSource { doc_id: string, page: number }
export interface RagEvidence { doc_id: string, page: number, text: string, bbox: number[] | null }
export interface RagCitation { doc_id: string, page: number, excerpt: string | null, bbox: number[] | null, source_kind: string }
export interface RagAnswerPart { text: string, status: 'supported' | 'unverified' | 'abstained', citations: RagCitation[] }
export interface RagTurn {
  id: string
  request_id: string
  document_id: string
  question: string
  answer: string
  mode: 'text' | 'hybrid'
  source_pages: RagSource[]
  evidence: RagEvidence[]
  answer_parts: RagAnswerPart[]
  retrieval: { mode: string, pages: number, latency_ms: number, source_pages: RagSource[] }
  created_at: string | null
}
export interface RagPin extends Omit<RagTurn, 'request_id' | 'mode' | 'retrieval'> { turn_id: string }

interface Pending { request_id: string, question: string, mode: 'text' | 'hybrid' }
function errorMessage(error: unknown, fallback: string): string {
  const value = error as { data?: { detail?: string }, message?: string }
  return value?.data?.detail || value?.message || fallback
}
interface WorkspaceState {
  documents: RagDocument[]
  turns: Record<string, RagTurn[]>
  pins: Record<string, RagPin[]>
  drafts: Record<string, string>
  modes: Record<string, 'text' | 'hybrid'>
  pending: Record<string, Pending | null>
  errors: Record<string, string | null>
  loaded: Record<string, boolean>
  busy: Record<string, boolean>
  pagePreview: { documentId: string, page: number, evidence: RagEvidence[], citation?: RagCitation } | null
}

export function useRagWorkspace() {
  const state = useState<WorkspaceState>('rag-workspace', () => ({
    documents: [], turns: {}, pins: {}, drafts: {}, modes: {}, pending: {}, errors: {}, loaded: {}, busy: {}, pagePreview: null
  }))
  const call = <T>(path: string, options?: Parameters<typeof $fetch<T>>[1]) => $fetch<T>(`/api/rag/${path}`, options)

  async function refreshDocuments() {
    try {
      state.value.documents = await call<RagDocument[]>('documents')
    } catch {
      // Keep the current document list visible when the backend is temporarily unavailable.
    }
  }
  async function loadDocument(documentId: string) {
    state.value.errors[documentId] = null
    try {
      const pending = getPending(documentId)
      const [turns, pins] = await Promise.all([
        call<RagTurn[]>(`documents/${documentId}/chat`), call<RagPin[]>(`documents/${documentId}/pins`)
      ])
      state.value.turns[documentId] = turns
      state.value.pins[documentId] = pins
      state.value.loaded[documentId] = true
      if (pending && turns.some(turn => turn.request_id === pending.request_id)) {
        state.value.pending[documentId] = null
        state.value.drafts[documentId] = ''
        if (import.meta.client) sessionStorage.removeItem(`rag-pending:${documentId}`)
      } else if (pending) {
        state.value.pending[documentId] = pending
        if (state.value.drafts[documentId] === undefined) state.value.drafts[documentId] = pending.question
      }
    } catch (error: unknown) {
      state.value.loaded[documentId] = false
      state.value.errors[documentId] = errorMessage(error, 'Could not load this document chat.')
    }
  }
  async function upload(file: File) {
    const form = new FormData()
    form.append('file', file)
    const doc = await call<RagDocument>('documents', { method: 'POST', body: form })
    state.value.documents.unshift(doc)
    return doc
  }
  async function processDocument(documentId: string) {
    const job = await call<{ id: string }>(`documents/${documentId}/process`, { method: 'POST' })
    await refreshDocuments()
    return job
  }
  function getPending(documentId: string): Pending | null {
    if (state.value.pending[documentId]) return state.value.pending[documentId]
    if (import.meta.client) {
      try {
        return JSON.parse(sessionStorage.getItem(`rag-pending:${documentId}`) || 'null')
      } catch {
        return null
      }
    }
    return null
  }
  async function send(documentId: string) {
    const question = (state.value.drafts[documentId] || '').trim()
    if (!question || state.value.busy[documentId] || !state.value.loaded[documentId]) return
    const pending = getPending(documentId) || {
      request_id: crypto.randomUUID(),
      question,
      mode: state.value.modes[documentId] || 'hybrid'
    }
    state.value.pending[documentId] = pending
    if (import.meta.client) sessionStorage.setItem(`rag-pending:${documentId}`, JSON.stringify(pending))
    state.value.busy[documentId] = true
    state.value.errors[documentId] = null
    try {
      const turn = await call<RagTurn>(`documents/${documentId}/chat/messages`, { method: 'POST', body: pending })
      if (turn.document_id !== documentId) throw new Error('The response belongs to a different document.')
      state.value.turns[documentId] = [...(state.value.turns[documentId] || []).filter(item => item.request_id !== turn.request_id), turn]
      state.value.drafts[documentId] = ''
      state.value.pending[documentId] = null
      sessionStorage.removeItem(`rag-pending:${documentId}`)
    } catch (error: unknown) {
      // Reconcile before offering retry in case the server committed before the connection failed.
      try {
        const history = await call<RagTurn[]>(`documents/${documentId}/chat`)
        state.value.turns[documentId] = history
        const committed = history.find(turn => turn.request_id === pending.request_id)
        if (committed) {
          state.value.drafts[documentId] = ''
          state.value.pending[documentId] = null
          sessionStorage.removeItem(`rag-pending:${documentId}`)
          return
        }
      } catch { /* preserve the request for retry */ }
      state.value.errors[documentId] = errorMessage(error, 'Could not send the question.')
    } finally {
      state.value.busy[documentId] = false
    }
  }
  async function pin(documentId: string, turnId: string) {
    const result = await call<RagPin>(`documents/${documentId}/pins`, { method: 'POST', body: { turn_id: turnId } })
    state.value.pins[documentId] = [result, ...(state.value.pins[documentId] || []).filter(pin => pin.turn_id !== turnId)]
  }
  async function unpin(documentId: string, pinId: string) {
    await call(`documents/${documentId}/pins/${pinId}`, { method: 'DELETE' })
    state.value.pins[documentId] = (state.value.pins[documentId] || []).filter(pin => pin.id !== pinId)
  }
  const documents = computed(() => state.value.documents)
  const turns = (id: string) => computed(() => state.value.turns[id] || [])
  const messages = (id: string) => computed<UIMessage[]>(() => (state.value.turns[id] || []).flatMap(turn => [
    { id: `${turn.id}-question`, role: 'user' as const, parts: [{ type: 'text' as const, text: turn.question }], createdAt: turn.created_at || undefined },
    { id: turn.id, role: 'assistant' as const, parts: [{ type: 'text' as const, text: turn.answer }], createdAt: turn.created_at || undefined }
  ]))

  return { state, documents, refreshDocuments, loadDocument, upload, processDocument, send, pin, unpin, turns, messages, call }
}
