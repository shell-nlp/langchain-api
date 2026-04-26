'use client'

import {
  ChangeEvent,
  KeyboardEvent,
  MouseEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import { marked } from 'marked'

import styles from './ChatInterface.module.css'

interface Message {
  id: string
  role: 'user' | 'ai'
  content: string
  reasoningContent?: string
  toolData?: ToolData[]
}

interface ToolData {
  toolCall: {
    id: string
    name: string
    args: Record<string, unknown>
  }
  toolOutput?: Array<{
    tool_call_id: string
    content: string
  }>
}

interface StreamEvent {
  event: 'token' | 'tool_calls' | 'tool_output' | '__interrupt__'
  data: {
    id?: string
    token?: string
    reasoning_token?: string
    tool_calls?: Array<{
      id: string
      name: string
      args: Record<string, unknown>
    }>
    tool_output?: Array<{
      tool_call_id: string
      content: unknown
    }>
    __interrupt__?: {
      action_requests: Array<{
        name: string
        description?: string
        args: Record<string, unknown>
      }>
    }
  }
}

interface KnowledgeBase {
  knowledge_base_id: string
  user_id: string
  name: string
  description: string
  index_prefix: string
  passage_index: string
  entity_index: string
  relation_index: string
  document_count: number
  chunk_count: number
  created_at: string
  updated_at: string
}

interface KnowledgeDocument {
  document_id: string
  knowledge_base_id: string
  user_id: string
  file_name: string
  display_name: string
  content_type: string
  file_size: number
  chunk_count: number
  storage_path: string
  created_at: string
  updated_at: string
}

interface UploadResult {
  knowledge_base: KnowledgeBase
  documents: KnowledgeDocument[]
  errors: Array<{
    file_name: string
    error: string
  }>
}

interface PaginatedKnowledgeBaseResponse {
  items: KnowledgeBase[]
  total: number
  page: number
  page_size: number
}

interface PaginatedKnowledgeDocumentResponse {
  items: KnowledgeDocument[]
  total: number
  page: number
  page_size: number
}

interface BulkDeleteKnowledgeBaseResponse {
  deleted_ids: string[]
  failed: Record<string, string>
}

interface BulkDeleteDocumentResponse {
  deleted_ids: string[]
  failed: Record<string, string>
  knowledge_base?: KnowledgeBase | null
}

type ViewMode = 'chat' | 'knowledge'
type KnowledgePage = 'libraries' | 'library-detail' | 'document-detail' | 'users'
type RequestMode = 'agent' | 'rag'

interface KnowledgeChunk {
  chunk_id: string
  document_id: string
  segment_id?: number | string | null
  content: string
  metadata: Record<string, unknown>
}

interface KnowledgeDocumentDetailResponse {
  knowledge_base: KnowledgeBase
  document: KnowledgeDocument
  chunks: KnowledgeChunk[]
  total_chunks: number
  page: number
  page_size: number
}

const DEFAULT_BACKEND_URL = 'http://localhost:7869'
const DEFAULT_AGENT_API_PATH = '/api/agent/general_api'
const DEFAULT_RAG_API_PATH = '/api/rag/general_api'
const KB_LIST_API_PATH = '/api/rag/knowledge-bases/list'
const KB_CREATE_API_PATH = '/api/rag/knowledge-bases/create'
const KB_DETAIL_API_PATH = '/api/rag/knowledge-bases/detail'
const KB_UPDATE_API_PATH = '/api/rag/knowledge-bases/update'
const KB_DELETE_API_PATH = '/api/rag/knowledge-bases/delete'
const KB_BULK_DELETE_API_PATH = '/api/rag/knowledge-bases/bulk-delete'
const KB_DOCUMENT_LIST_API_PATH = '/api/rag/knowledge-bases/documents/list'
const KB_DOCUMENT_DETAIL_API_PATH = '/api/rag/knowledge-bases/documents/detail'
const KB_DOCUMENT_UPLOAD_API_PATH = '/api/rag/knowledge-bases/documents/upload'
const KB_DOCUMENT_UPDATE_API_PATH = '/api/rag/knowledge-bases/documents/update'
const KB_DOCUMENT_DELETE_API_PATH = '/api/rag/knowledge-bases/documents/delete'
const KB_DOCUMENT_BULK_DELETE_API_PATH =
  '/api/rag/knowledge-bases/documents/bulk-delete'
const KNOWLEDGE_BASE_PAGE_SIZE = 8
const DOCUMENT_PAGE_SIZE = 10
const DEFAULT_KNOWLEDGE_PAGE: KnowledgePage = 'libraries'

function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL
  if (envUrl) return envUrl.replace(/\/$/, '')

  if (typeof window === 'undefined') return ''

  const { hostname, origin } = window.location
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return DEFAULT_BACKEND_URL
  }

  return origin
}

function getApiUrl(path: string): string {
  return `${getApiBaseUrl()}${path}`
}

function normalizeKnowledgePage(value?: string): KnowledgePage {
  switch (value) {
    case 'library-detail':
    case 'document-detail':
    case 'users':
      return value
    default:
      return 'libraries'
  }
}

function getRouteHash(
  viewMode: ViewMode,
  knowledgePage: KnowledgePage = DEFAULT_KNOWLEDGE_PAGE
): string {
  return viewMode === 'chat' ? '#/chat' : `#/knowledge/${knowledgePage}`
}

function parseRouteHash(hash: string): {
  viewMode: ViewMode
  knowledgePage: KnowledgePage
} {
  const normalized = hash.replace(/^#/, '').replace(/^\/+/, '')
  const parts = normalized.split('/').filter(Boolean)

  if (parts[0] === 'knowledge') {
    return {
      viewMode: 'knowledge',
      knowledgePage: normalizeKnowledgePage(parts[1]),
    }
  }

  return {
    viewMode: 'chat',
    knowledgePage: DEFAULT_KNOWLEDGE_PAGE,
  }
}

function generateSessionId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

function formatDateTime(value: string): string {
  try {
    return new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value))
  } catch {
    return value
  }
}

function getPageTotal(total: number, pageSize: number): number {
  return Math.max(1, Math.ceil(total / pageSize))
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `HTTP ${response.status}`
    try {
      const payload = await response.json()
      if (payload?.detail) {
        message = String(payload.detail)
      }
    } catch {
      // ignore parse error
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

function getToolIcon(toolName: string): string {
  const iconMap: Record<string, string> = {
    search: 'S',
    calculator: 'M',
    calc: 'M',
    math: 'M',
    weather: 'W',
    time: 'T',
    date: 'D',
    file: 'F',
    read: 'R',
    write: 'W',
    edit: 'E',
    api: 'A',
    http: 'A',
    request: 'A',
    fetch: 'A',
    python: 'P',
    code: 'C',
    exec: 'C',
    run: 'R',
    bash: 'B',
    git: 'G',
    translate: 'TR',
    analyze: 'AN',
    browser: 'BR',
  }
  const lower = toolName.toLowerCase()
  for (const [key, icon] of Object.entries(iconMap)) {
    if (lower.includes(key)) return icon
  }
  return 'TL'
}

export default function ChatInterface() {
  const [viewMode, setViewMode] = useState<ViewMode>('chat')
  const [knowledgePage, setKnowledgePage] =
    useState<KnowledgePage>(DEFAULT_KNOWLEDGE_PAGE)

  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [status, setStatus] = useState<'ready' | 'connecting' | 'error'>('ready')
  const [isProcessing, setIsProcessing] = useState(false)
  const [internetSearch, setInternetSearch] = useState(false)
  const [deepThinking, setDeepThinking] = useState(false)
  const [useKnowledgeBase, setUseKnowledgeBase] = useState(false)
  const [showInterrupt, setShowInterrupt] = useState(false)
  const [interruptData, setInterruptData] = useState<StreamEvent['data']['__interrupt__'] | null>(null)

  const [userId, setUserId] = useState('demo-user')
  const [userIdDraft, setUserIdDraft] = useState('demo-user')

  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [knowledgeBaseTotal, setKnowledgeBaseTotal] = useState(0)
  const [knowledgeBasePage, setKnowledgeBasePage] = useState(1)
  const [knowledgeBaseSearchInput, setKnowledgeBaseSearchInput] = useState('')
  const [knowledgeBaseSearch, setKnowledgeBaseSearch] = useState('')

  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState('')
  const [selectedKnowledgeBase, setSelectedKnowledgeBase] = useState<KnowledgeBase | null>(null)
  const [selectedKnowledgeBaseName, setSelectedKnowledgeBaseName] = useState('')
  const [selectedKnowledgeBaseDescription, setSelectedKnowledgeBaseDescription] = useState('')
  const [checkedKnowledgeBaseIds, setCheckedKnowledgeBaseIds] = useState<string[]>([])

  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [documentTotal, setDocumentTotal] = useState(0)
  const [documentPage, setDocumentPage] = useState(1)
  const [documentSearchInput, setDocumentSearchInput] = useState('')
  const [documentSearch, setDocumentSearch] = useState('')
  const [checkedDocumentIds, setCheckedDocumentIds] = useState<string[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState('')
  const [selectedDocumentDetail, setSelectedDocumentDetail] =
    useState<KnowledgeDocumentDetailResponse | null>(null)
  const [documentChunkPage, setDocumentChunkPage] = useState(1)
  const [loadingDocumentDetail, setLoadingDocumentDetail] = useState(false)

  const [knowledgeBaseName, setKnowledgeBaseName] = useState('')
  const [knowledgeBaseDescription, setKnowledgeBaseDescription] = useState('')
  const [showCreateKnowledgeBaseModal, setShowCreateKnowledgeBaseModal] = useState(false)
  const [savedUsers, setSavedUsers] = useState<string[]>([])

  const [managementError, setManagementError] = useState('')
  const [managementNotice, setManagementNotice] = useState('')
  const [loadingKnowledgeBases, setLoadingKnowledgeBases] = useState(false)
  const [loadingKnowledgeBaseDetail, setLoadingKnowledgeBaseDetail] = useState(false)
  const [loadingDocuments, setLoadingDocuments] = useState(false)
  const [savingKnowledgeBase, setSavingKnowledgeBase] = useState(false)
  const [uploadingDocuments, setUploadingDocuments] = useState(false)
  const [deletingBulk, setDeletingBulk] = useState(false)

  const chatContainerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const uploadInputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentAssistantMessageIdRef = useRef<string | null>(null)
  const processedToolCallIdsRef = useRef<Set<string>>(new Set())
  const requestModeRef = useRef<RequestMode>('agent')
  const requestKnowledgeBaseRef = useRef<KnowledgeBase | null>(null)

  const scrollToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [])

  const clearChat = useCallback(() => {
    setMessages([])
    currentAssistantMessageIdRef.current = null
    processedToolCallIdsRef.current.clear()
    const newId = generateSessionId()
    localStorage.setItem('rag_chat_session_id', newId)
    setSessionId(newId)
  }, [])

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => {
      const existing = prev.find((item) => item.id === message.id)
      if (existing) {
        return prev.map((item) => (item.id === message.id ? { ...item, ...message } : item))
      }
      return [...prev, message]
    })
  }, [])

  const ensureAssistantMessage = useCallback(() => {
    let assistantMessageId = currentAssistantMessageIdRef.current
    if (!assistantMessageId) {
      assistantMessageId = generateMessageId()
      currentAssistantMessageIdRef.current = assistantMessageId
      addMessage({ id: assistantMessageId, role: 'ai', content: '', toolData: [] })
    }
    return assistantMessageId
  }, [addMessage])

  const updateAssistantMessage = useCallback(
    (updater: (message: Message) => Message) => {
      const assistantMessageId = ensureAssistantMessage()
      setMessages((prev) =>
        prev.map((item) => (item.id === assistantMessageId ? updater(item) : item))
      )
    },
    [ensureAssistantMessage]
  )

  useEffect(() => {
    if (typeof window === 'undefined') return

    const storedUserId = localStorage.getItem('rag_user_id') || 'demo-user'
    const storedSessionId =
      localStorage.getItem('rag_chat_session_id') || generateSessionId()
    const storedUsers = localStorage.getItem('rag_saved_users')
    const normalizedUsers = Array.from(
      new Set(
        [storedUserId, ...(storedUsers ? JSON.parse(storedUsers) : [])].filter(Boolean)
      )
    )

    localStorage.setItem('rag_user_id', storedUserId)
    localStorage.setItem('rag_chat_session_id', storedSessionId)
    localStorage.setItem('rag_saved_users', JSON.stringify(normalizedUsers))

    setUserId(storedUserId)
    setUserIdDraft(storedUserId)
    setSessionId(storedSessionId)
    setSavedUsers(normalizedUsers)
  }, [])

  const navigateTo = useCallback(
    (
      nextViewMode: ViewMode,
      nextKnowledgePage: KnowledgePage = DEFAULT_KNOWLEDGE_PAGE,
      replace = false
    ) => {
      const safeKnowledgePage =
        (
          nextKnowledgePage === 'library-detail' ||
          nextKnowledgePage === 'document-detail'
        ) &&
        !selectedKnowledgeBase
          ? 'libraries'
          : nextKnowledgePage

      setViewMode(nextViewMode)
      setKnowledgePage(safeKnowledgePage)

      if (typeof window === 'undefined') return
      const nextHash = getRouteHash(nextViewMode, safeKnowledgePage)
      if (window.location.hash === nextHash) return

      if (replace) {
        window.history.replaceState(null, '', nextHash)
      } else {
        window.history.pushState(null, '', nextHash)
      }
    },
    [selectedKnowledgeBase]
  )

  useEffect(() => {
    if (typeof window === 'undefined') return

    const syncRoute = () => {
      const route = parseRouteHash(window.location.hash)
      const safeKnowledgePage =
        (
          route.knowledgePage === 'library-detail' ||
          route.knowledgePage === 'document-detail'
        ) &&
        !selectedKnowledgeBase
          ? 'libraries'
          : route.knowledgePage
      setViewMode(route.viewMode)
      setKnowledgePage(safeKnowledgePage)

      const expectedHash = getRouteHash(route.viewMode, safeKnowledgePage)
      if (window.location.hash !== expectedHash) {
        window.history.replaceState(null, '', expectedHash)
      }
    }

    if (!window.location.hash) {
      window.history.replaceState(null, '', getRouteHash('chat'))
    }

    syncRoute()
    window.addEventListener('hashchange', syncRoute)
    return () => window.removeEventListener('hashchange', syncRoute)
  }, [selectedKnowledgeBase])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const escapeHtml = (text: string): string => {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  const parseMarkdown = (text: string): string => {
    try {
      return marked.parse(text, { async: false }) as string
    } catch {
      return escapeHtml(text)
    }
  }

  const stringifyToolContent = (content: unknown): string => {
    if (typeof content === 'string') return content
    if (content == null) return ''
    try {
      return JSON.stringify(content, null, 2)
    } catch {
      return String(content)
    }
  }

  const loadKnowledgeBases = useCallback(
    async (targetUserId: string, page = knowledgeBasePage, search = knowledgeBaseSearch) => {
      setLoadingKnowledgeBases(true)
      setManagementError('')

      try {
        const result = await fetchJson<PaginatedKnowledgeBaseResponse>(
          getApiUrl(KB_LIST_API_PATH),
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: targetUserId,
              search,
              page,
              page_size: KNOWLEDGE_BASE_PAGE_SIZE,
            }),
          }
        )
        setKnowledgeBases(result.items)
        setKnowledgeBaseTotal(result.total)

        if (selectedKnowledgeBaseId) {
          const matched = result.items.find(
            (item) => item.knowledge_base_id === selectedKnowledgeBaseId
          )
          if (matched) {
            setSelectedKnowledgeBase(matched)
            setSelectedKnowledgeBaseName(matched.name)
            setSelectedKnowledgeBaseDescription(matched.description)
          } else {
            setSelectedKnowledgeBaseId('')
            setSelectedKnowledgeBase(null)
          }
        }
      } catch (error) {
        setManagementError(
          error instanceof Error ? error.message : 'Failed to load knowledge bases.'
        )
        setKnowledgeBases([])
        setKnowledgeBaseTotal(0)
      } finally {
        setLoadingKnowledgeBases(false)
      }
    },
    [knowledgeBasePage, knowledgeBaseSearch, selectedKnowledgeBaseId]
  )

  const loadKnowledgeBaseDetail = useCallback(async (targetUserId: string, knowledgeBaseId: string) => {
    if (!knowledgeBaseId) return
    setLoadingKnowledgeBaseDetail(true)
    setManagementError('')
    try {
      const result = await fetchJson<KnowledgeBase>(getApiUrl(KB_DETAIL_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: targetUserId,
          knowledge_base_id: knowledgeBaseId,
        }),
      })
      setSelectedKnowledgeBase(result)
      setSelectedKnowledgeBaseId(result.knowledge_base_id)
      setSelectedKnowledgeBaseName(result.name)
      setSelectedKnowledgeBaseDescription(result.description)
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to load knowledge base.'
      )
    } finally {
      setLoadingKnowledgeBaseDetail(false)
    }
  }, [])

  const loadDocuments = useCallback(
    async (
      targetUserId: string,
      knowledgeBaseId: string,
      page = documentPage,
      search = documentSearch
    ) => {
      if (!knowledgeBaseId) {
        setDocuments([])
        setDocumentTotal(0)
        return
      }

      setLoadingDocuments(true)
      setManagementError('')
      try {
        const result = await fetchJson<PaginatedKnowledgeDocumentResponse>(
          getApiUrl(KB_DOCUMENT_LIST_API_PATH),
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: targetUserId,
              knowledge_base_id: knowledgeBaseId,
              search,
              page,
              page_size: DOCUMENT_PAGE_SIZE,
            }),
          }
        )
        setDocuments(result.items)
        setDocumentTotal(result.total)
      } catch (error) {
        setManagementError(
          error instanceof Error ? error.message : 'Failed to load documents.'
        )
        setDocuments([])
        setDocumentTotal(0)
      } finally {
        setLoadingDocuments(false)
      }
    },
    [documentPage, documentSearch]
  )

  const loadDocumentDetail = useCallback(
    async (
      targetUserId: string,
      knowledgeBaseId: string,
      documentId: string,
      page = documentChunkPage
    ) => {
      if (!knowledgeBaseId || !documentId) {
        setSelectedDocumentDetail(null)
        return
      }

      setLoadingDocumentDetail(true)
      setManagementError('')
      try {
        const result = await fetchJson<KnowledgeDocumentDetailResponse>(
          getApiUrl(KB_DOCUMENT_DETAIL_API_PATH),
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: targetUserId,
              knowledge_base_id: knowledgeBaseId,
              document_id: documentId,
              page,
              page_size: 8,
            }),
          }
        )
        setSelectedDocumentDetail(result)
      } catch (error) {
        setManagementError(
          error instanceof Error ? error.message : 'Failed to load document detail.'
        )
        setSelectedDocumentDetail(null)
      } finally {
        setLoadingDocumentDetail(false)
      }
    },
    [documentChunkPage]
  )

  useEffect(() => {
    if (!userId) return
    void loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
  }, [knowledgeBasePage, knowledgeBaseSearch, loadKnowledgeBases, userId])

  useEffect(() => {
    if (!selectedKnowledgeBaseId) {
      setSelectedKnowledgeBase(null)
      setDocuments([])
      setDocumentTotal(0)
      return
    }
    void loadKnowledgeBaseDetail(userId, selectedKnowledgeBaseId)
  }, [loadKnowledgeBaseDetail, selectedKnowledgeBaseId, userId])

  useEffect(() => {
    if (!selectedKnowledgeBaseId) return
    void loadDocuments(userId, selectedKnowledgeBaseId, documentPage, documentSearch)
  }, [documentPage, documentSearch, loadDocuments, selectedKnowledgeBaseId, userId])

  useEffect(() => {
    if (!selectedKnowledgeBaseId || !selectedDocumentId || knowledgePage !== 'document-detail') {
      return
    }
    void loadDocumentDetail(
      userId,
      selectedKnowledgeBaseId,
      selectedDocumentId,
      documentChunkPage
    )
  }, [
    documentChunkPage,
    knowledgePage,
    loadDocumentDetail,
    selectedDocumentId,
    selectedKnowledgeBaseId,
    userId,
  ])

  const selectKnowledgeBase = (knowledgeBase: KnowledgeBase) => {
    if (useKnowledgeBase && knowledgeBase.knowledge_base_id !== selectedKnowledgeBaseId) {
      clearChat()
      setCheckedDocumentIds([])
    }
    setSelectedKnowledgeBaseId(knowledgeBase.knowledge_base_id)
    setSelectedKnowledgeBase(knowledgeBase)
    setSelectedKnowledgeBaseName(knowledgeBase.name)
    setSelectedKnowledgeBaseDescription(knowledgeBase.description)
    setDocumentPage(1)
    setDocumentSearch('')
    setDocumentSearchInput('')
    setSelectedDocumentId('')
    setSelectedDocumentDetail(null)
    setDocumentChunkPage(1)
  }

  const handleKnowledgeBaseToggle = (checked: boolean) => {
    if (checked !== useKnowledgeBase) {
      clearChat()
      setShowInterrupt(false)
      setInterruptData(null)
      requestModeRef.current = checked ? 'rag' : 'agent'
      requestKnowledgeBaseRef.current = checked ? selectedKnowledgeBase : null
    }
    setUseKnowledgeBase(checked)
  }

  const persistSavedUsers = (nextUsers: string[]) => {
    const normalized = Array.from(new Set(nextUsers.map((item) => item.trim()).filter(Boolean)))
    setSavedUsers(normalized)
    if (typeof window !== 'undefined') {
      localStorage.setItem('rag_saved_users', JSON.stringify(normalized))
    }
  }

  const switchUser = (nextUserId: string) => {
    const normalizedUserId = nextUserId.trim() || 'demo-user'
    if (typeof window !== 'undefined') {
      localStorage.setItem('rag_user_id', normalizedUserId)
    }
    setUserId(normalizedUserId)
    setUserIdDraft(normalizedUserId)
    setKnowledgeBasePage(1)
    setKnowledgeBaseSearch('')
    setKnowledgeBaseSearchInput('')
    setDocumentPage(1)
    setDocumentSearch('')
    setDocumentSearchInput('')
    setSelectedKnowledgeBaseId('')
    setSelectedKnowledgeBase(null)
    setSelectedDocumentId('')
    setSelectedDocumentDetail(null)
    setKnowledgeBases([])
    setDocuments([])
    setCheckedKnowledgeBaseIds([])
    setCheckedDocumentIds([])
    persistSavedUsers([normalizedUserId, ...savedUsers])
    setManagementNotice(`Active user switched to ${normalizedUserId}.`)
    setManagementError('')
    clearChat()
    navigateTo('knowledge', 'users')
  }

  const applyUserId = () => {
    switchUser(userIdDraft)
  }

  const removeSavedUser = (targetUserId: string) => {
    const nextUsers = savedUsers.filter((item) => item !== targetUserId)
    persistSavedUsers(nextUsers.length > 0 ? nextUsers : ['demo-user'])
  }

  const openKnowledgeBaseLibrary = (knowledgeBase: KnowledgeBase) => {
    selectKnowledgeBase(knowledgeBase)
    navigateTo('knowledge', 'library-detail')
  }

  const openDocumentDetail = (document: KnowledgeDocument) => {
    setSelectedDocumentId(document.document_id)
    setDocumentChunkPage(1)
    navigateTo('knowledge', 'document-detail')
  }

  const createKnowledgeBase = async () => {
    const name = knowledgeBaseName.trim()
    if (!name) {
      setManagementError('Knowledge base name is required.')
      return
    }

    setSavingKnowledgeBase(true)
    setManagementError('')
    setManagementNotice('')

    try {
      const created = await fetchJson<KnowledgeBase>(getApiUrl(KB_CREATE_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          name,
          description: knowledgeBaseDescription.trim(),
        }),
      })

      setKnowledgeBasePage(1)
      await loadKnowledgeBases(userId, 1, knowledgeBaseSearch)
      selectKnowledgeBase(created)
      navigateTo('knowledge', 'library-detail')
      setKnowledgeBaseName('')
      setKnowledgeBaseDescription('')
      setShowCreateKnowledgeBaseModal(false)
      setManagementNotice(`Knowledge base "${created.name}" created.`)
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to create knowledge base.'
      )
    } finally {
      setSavingKnowledgeBase(false)
    }
  }

  const saveKnowledgeBase = async () => {
    if (!selectedKnowledgeBase) return

    setSavingKnowledgeBase(true)
    setManagementError('')
    setManagementNotice('')

    try {
      const updated = await fetchJson<KnowledgeBase>(getApiUrl(KB_UPDATE_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          knowledge_base_id: selectedKnowledgeBase.knowledge_base_id,
          name: selectedKnowledgeBaseName.trim(),
          description: selectedKnowledgeBaseDescription.trim(),
        }),
      })
      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      selectKnowledgeBase(updated)
      setManagementNotice(`Knowledge base "${updated.name}" updated.`)
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to update knowledge base.'
      )
    } finally {
      setSavingKnowledgeBase(false)
    }
  }

  const deleteKnowledgeBase = async (knowledgeBaseId?: string) => {
    const targetId = knowledgeBaseId || selectedKnowledgeBase?.knowledge_base_id
    const targetName =
      knowledgeBases.find((item) => item.knowledge_base_id === targetId)?.name ||
      selectedKnowledgeBase?.name ||
      'knowledge base'

    if (!targetId) return
    if (!window.confirm(`Delete knowledge base "${targetName}" and all Elasticsearch data?`)) {
      return
    }

    setSavingKnowledgeBase(true)
    setManagementError('')
    setManagementNotice('')

    try {
      await fetchJson<unknown>(getApiUrl(KB_DELETE_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          knowledge_base_id: targetId,
        }),
      })

      if (targetId === selectedKnowledgeBaseId) {
        setSelectedKnowledgeBaseId('')
        setSelectedKnowledgeBase(null)
        setSelectedKnowledgeBaseName('')
        setSelectedKnowledgeBaseDescription('')
        setSelectedDocumentId('')
        setSelectedDocumentDetail(null)
        setDocuments([])
        setDocumentTotal(0)
        navigateTo('knowledge', 'libraries')
      }
      setCheckedKnowledgeBaseIds((prev) => prev.filter((item) => item !== targetId))
      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      setManagementNotice(`Knowledge base "${targetName}" deleted.`)
      clearChat()
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to delete knowledge base.'
      )
    } finally {
      setSavingKnowledgeBase(false)
    }
  }

  const bulkDeleteKnowledgeBases = async () => {
    if (checkedKnowledgeBaseIds.length === 0) return
    if (
      !window.confirm(
        `Delete ${checkedKnowledgeBaseIds.length} selected knowledge base(s) and all Elasticsearch data?`
      )
    ) {
      return
    }

    setDeletingBulk(true)
    setManagementError('')
    setManagementNotice('')

    try {
      const result = await fetchJson<BulkDeleteKnowledgeBaseResponse>(
        getApiUrl(KB_BULK_DELETE_API_PATH),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            knowledge_base_ids: checkedKnowledgeBaseIds,
          }),
        }
      )
      if (result.deleted_ids.includes(selectedKnowledgeBaseId)) {
        setSelectedKnowledgeBaseId('')
        setSelectedKnowledgeBase(null)
        setSelectedKnowledgeBaseName('')
        setSelectedKnowledgeBaseDescription('')
        setSelectedDocumentId('')
        setSelectedDocumentDetail(null)
        setDocuments([])
        setDocumentTotal(0)
        clearChat()
        navigateTo('knowledge', 'libraries')
      }
      setCheckedKnowledgeBaseIds([])
      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      setManagementNotice(
        `Deleted ${result.deleted_ids.length} knowledge base(s).`
      )
      if (Object.keys(result.failed).length > 0) {
        setManagementError(
          Object.entries(result.failed)
            .map(([id, message]) => `${id}: ${message}`)
            .join('\n')
        )
      }
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to bulk delete knowledge bases.'
      )
    } finally {
      setDeletingBulk(false)
    }
  }

  const openUploadDialog = () => {
    uploadInputRef.current?.click()
  }

  const handleUploadFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (!selectedKnowledgeBase || files.length === 0) return

    setUploadingDocuments(true)
    setManagementError('')
    setManagementNotice('')

    const formData = new FormData()
    formData.append('user_id', userId)
    formData.append('knowledge_base_id', selectedKnowledgeBase.knowledge_base_id)
    files.forEach((file) => formData.append('files', file))

    try {
      const result = await fetchJson<UploadResult>(
        getApiUrl(KB_DOCUMENT_UPLOAD_API_PATH),
        {
          method: 'POST',
          body: formData,
        }
      )

      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      await loadKnowledgeBaseDetail(userId, selectedKnowledgeBase.knowledge_base_id)
      await loadDocuments(userId, selectedKnowledgeBase.knowledge_base_id, 1, documentSearch)
      setDocumentPage(1)

      const successCount = result.documents.length
      const errorCount = result.errors.length
      setManagementNotice(
        errorCount
          ? `${successCount} file(s) indexed, ${errorCount} failed.`
          : `${successCount} file(s) indexed successfully.`
      )
      if (errorCount) {
        setManagementError(
          result.errors.map((item) => `${item.file_name}: ${item.error}`).join('\n')
        )
      }
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to upload files.'
      )
    } finally {
      setUploadingDocuments(false)
      event.target.value = ''
    }
  }

  const renameDocument = async (document: KnowledgeDocument) => {
    if (!selectedKnowledgeBase) return
    const nextName = window.prompt('Document display name', document.display_name)
    if (!nextName) return

    setManagementError('')
    setManagementNotice('')
    try {
      await fetchJson<KnowledgeDocument>(getApiUrl(KB_DOCUMENT_UPDATE_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          knowledge_base_id: selectedKnowledgeBase.knowledge_base_id,
          document_id: document.document_id,
          display_name: nextName,
        }),
      })
      await loadDocuments(
        userId,
        selectedKnowledgeBase.knowledge_base_id,
        documentPage,
        documentSearch
      )
      if (selectedDocumentId === document.document_id) {
        await loadDocumentDetail(
          userId,
          selectedKnowledgeBase.knowledge_base_id,
          document.document_id,
          documentChunkPage
        )
      }
      setManagementNotice(`Document "${nextName}" updated.`)
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to update document.'
      )
    }
  }

  const deleteDocument = async (documentId?: string, documentName?: string) => {
    if (!selectedKnowledgeBase) return
    const targetId = documentId
    if (!targetId) return
    const targetName =
      documentName ||
      documents.find((item) => item.document_id === targetId)?.display_name ||
      'document'
    if (!window.confirm(`Delete document "${targetName}" from Elasticsearch?`)) return

    setManagementError('')
    setManagementNotice('')
    try {
      await fetchJson<unknown>(getApiUrl(KB_DOCUMENT_DELETE_API_PATH), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          knowledge_base_id: selectedKnowledgeBase.knowledge_base_id,
          document_id: targetId,
        }),
      })
      setCheckedDocumentIds((prev) => prev.filter((item) => item !== targetId))
      if (selectedDocumentId === targetId) {
        setSelectedDocumentId('')
        setSelectedDocumentDetail(null)
        navigateTo('knowledge', 'library-detail')
      }
      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      await loadKnowledgeBaseDetail(userId, selectedKnowledgeBase.knowledge_base_id)
      await loadDocuments(
        userId,
        selectedKnowledgeBase.knowledge_base_id,
        documentPage,
        documentSearch
      )
      setManagementNotice(`Document "${targetName}" deleted from Elasticsearch.`)
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to delete document.'
      )
    }
  }

  const bulkDeleteDocuments = async () => {
    if (!selectedKnowledgeBase || checkedDocumentIds.length === 0) return
    if (
      !window.confirm(
        `Delete ${checkedDocumentIds.length} selected document(s) from Elasticsearch?`
      )
    ) {
      return
    }

    setDeletingBulk(true)
    setManagementError('')
    setManagementNotice('')
    try {
      const result = await fetchJson<BulkDeleteDocumentResponse>(
        getApiUrl(KB_DOCUMENT_BULK_DELETE_API_PATH),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            knowledge_base_id: selectedKnowledgeBase.knowledge_base_id,
            document_ids: checkedDocumentIds,
          }),
        }
      )
      setCheckedDocumentIds([])
      if (checkedDocumentIds.includes(selectedDocumentId)) {
        setSelectedDocumentId('')
        setSelectedDocumentDetail(null)
        navigateTo('knowledge', 'library-detail')
      }
      await loadKnowledgeBases(userId, knowledgeBasePage, knowledgeBaseSearch)
      await loadKnowledgeBaseDetail(userId, selectedKnowledgeBase.knowledge_base_id)
      await loadDocuments(
        userId,
        selectedKnowledgeBase.knowledge_base_id,
        documentPage,
        documentSearch
      )
      setManagementNotice(`Deleted ${result.deleted_ids.length} document(s).`)
      if (Object.keys(result.failed).length > 0) {
        setManagementError(
          Object.entries(result.failed)
            .map(([id, message]) => `${id}: ${message}`)
            .join('\n')
        )
      }
    } catch (error) {
      setManagementError(
        error instanceof Error ? error.message : 'Failed to bulk delete documents.'
      )
    } finally {
      setDeletingBulk(false)
    }
  }

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      const data = event.data
      if (!data) return

      switch (event.event) {
        case 'token': {
          if (data.reasoning_token) {
            updateAssistantMessage((message) => ({
              ...message,
              reasoningContent: `${message.reasoningContent || ''}${data.reasoning_token}`,
            }))
          }

          if (data.token) {
            updateAssistantMessage((message) => ({
              ...message,
              content: `${message.content}${data.token}`,
            }))
          }
          break
        }

        case 'tool_calls': {
          if (data.tool_calls?.length) {
            updateAssistantMessage((message) => {
              const tools = [...(message.toolData || [])]
              for (const toolCall of data.tool_calls || []) {
                if (!tools.some((tool) => tool.toolCall.id === toolCall.id)) {
                  tools.push({ toolCall, toolOutput: [] })
                }
              }
              return { ...message, toolData: tools }
            })
          }
          break
        }

        case 'tool_output': {
          if (data.tool_output?.length) {
            updateAssistantMessage((message) => {
              let tools = [...(message.toolData || [])]

              for (const output of data.tool_output || []) {
                if (processedToolCallIdsRef.current.has(output.tool_call_id)) {
                  continue
                }
                processedToolCallIdsRef.current.add(output.tool_call_id)

                const normalizedOutput = {
                  ...output,
                  content: stringifyToolContent(output.content),
                }
                const existingToolIndex = tools.findIndex(
                  (tool) => tool.toolCall.id === output.tool_call_id
                )

                if (existingToolIndex >= 0) {
                  const existingTool = tools[existingToolIndex]
                  tools = tools.map((tool, index) =>
                    index === existingToolIndex
                      ? {
                          ...existingTool,
                          toolOutput: [
                            ...(existingTool.toolOutput || []),
                            normalizedOutput,
                          ],
                        }
                      : tool
                  )
                } else {
                  tools.push({
                    toolCall: {
                      id: output.tool_call_id,
                      name: 'tool',
                      args: {},
                    },
                    toolOutput: [normalizedOutput],
                  })
                }
              }

              return { ...message, toolData: tools }
            })
          }
          break
        }

        case '__interrupt__': {
          if (data.__interrupt__) {
            setInterruptData(data.__interrupt__)
            setShowInterrupt(true)
            setIsProcessing(false)
            setStatus('ready')
          }
          break
        }
      }
    },
    [updateAssistantMessage]
  )

  const readEventStream = useCallback(
    async (response: Response) => {
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      const processChunk = (chunk: string) => {
        const normalized = chunk.replace(/\r\n/g, '\n')
        const parts = normalized.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const dataLines = part
            .split('\n')
            .map((line) => line.trim())
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trim())

          if (dataLines.length === 0) continue

          try {
            handleStreamEvent(JSON.parse(dataLines.join('\n')) as StreamEvent)
          } catch (error) {
            console.error('Parse error:', error, dataLines.join('\n'))
          }
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        processChunk(buffer)
      }

      buffer += decoder.decode()
      if (buffer.trim()) {
        processChunk(`${buffer}\n\n`)
      }
    },
    [handleStreamEvent]
  )

  const sendMessage = async () => {
    const query = inputValue.trim()
    if (!query || isProcessing || !sessionId) return

    const requestMode: RequestMode = useKnowledgeBase ? 'rag' : 'agent'
    if (requestMode === 'rag' && !selectedKnowledgeBase) {
      setManagementError('启用知识库后，必须先选择一个知识库。')
      navigateTo('knowledge', 'libraries')
      return
    }

    addMessage({
      id: generateMessageId(),
      role: 'user',
      content: query,
    })
    setInputValue('')

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    setIsProcessing(true)
    setStatus('connecting')

    const assistantMessageId = generateMessageId()
    currentAssistantMessageIdRef.current = assistantMessageId
    processedToolCallIdsRef.current.clear()
    requestModeRef.current = requestMode
    requestKnowledgeBaseRef.current = selectedKnowledgeBase
    addMessage({ id: assistantMessageId, role: 'ai', content: '', toolData: [] })

    abortControllerRef.current = new AbortController()

    try {
      const payload: Record<string, unknown> = {
        query,
        session_id: sessionId,
        user_id: userId,
        internet_search: internetSearch,
        deep_thinking: deepThinking,
      }
      if (requestMode === 'rag' && selectedKnowledgeBase) {
        payload.index_name = selectedKnowledgeBase.passage_index
        payload.graph_name = selectedKnowledgeBase.index_prefix
      }

      const response = await fetch(
        getApiUrl(
          requestMode === 'rag' ? DEFAULT_RAG_API_PATH : DEFAULT_AGENT_API_PATH
        ),
        {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal,
      }
      )

      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      await readEventStream(response)
      setStatus('ready')
    } catch (error: unknown) {
      if (error instanceof Error && error.name !== 'AbortError') {
        setStatus('error')
        addMessage({
          id: generateMessageId(),
          role: 'ai',
          content: `Request failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        })
      } else {
        setStatus('ready')
      }
    } finally {
      setIsProcessing(false)
      currentAssistantMessageIdRef.current = null
      processedToolCallIdsRef.current.clear()
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void sendMessage()
    }
  }

  const abortRequest = () => {
    abortControllerRef.current?.abort()
  }

  const handleInterruptAction = async (decision: 'approve' | 'reject' | 'edit') => {
    if (!interruptData) return

    const requestMode = requestModeRef.current
    const requestKnowledgeBase = requestKnowledgeBaseRef.current
    if (requestMode === 'rag' && !requestKnowledgeBase) {
      setManagementError('当前中断来自知识库问答，但未找到对应知识库，请重新发起知识库对话。')
      setShowInterrupt(false)
      setInterruptData(null)
      return
    }

    let editedActions: Array<{ name: string; args: Record<string, unknown> }> | undefined
    if (decision === 'edit') {
      try {
        editedActions = interruptData.action_requests.map((action, index) => {
          const editor = document.getElementById(
            `argsEditor${index}`
          ) as HTMLTextAreaElement | null
          return {
            name: action.name,
            args: editor ? JSON.parse(editor.value) : action.args,
          }
        })
      } catch {
        setManagementError('Interrupt arguments must be valid JSON.')
        return
      }
    }

    setShowInterrupt(false)
    addMessage({
      id: generateMessageId(),
      role: 'user',
      content:
        decision === 'approve'
          ? 'Approve pending action.'
          : decision === 'reject'
            ? 'Reject pending action.'
            : 'Resume with edited arguments.',
    })

    setIsProcessing(true)
    setStatus('connecting')
    abortControllerRef.current = new AbortController()

    try {
      const decisions = (interruptData.action_requests || []).map((action, index) => {
        if (decision === 'edit' && editedActions?.[index]) {
          return {
            type: 'edit',
            edited_action: editedActions[index],
          }
        }
        return { type: decision, message: `Decision applied to ${action.name}.` }
      })

      const payload: Record<string, unknown> = {
        resume: { decisions },
        session_id: sessionId,
        user_id: userId,
      }
      if (requestMode === 'rag' && requestKnowledgeBase) {
        payload.index_name = requestKnowledgeBase.passage_index
        payload.graph_name = requestKnowledgeBase.index_prefix
      }

      const response = await fetch(
        getApiUrl(
          requestMode === 'rag' ? DEFAULT_RAG_API_PATH : DEFAULT_AGENT_API_PATH
        ),
        {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal,
      }
      )

      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      await readEventStream(response)
      setStatus('ready')
    } catch (error: unknown) {
      if (error instanceof Error && error.name !== 'AbortError') {
        setStatus('error')
        addMessage({
          id: generateMessageId(),
          role: 'ai',
          content: `Resume failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        })
      }
    } finally {
      setIsProcessing(false)
      setInterruptData(null)
      processedToolCallIdsRef.current.clear()
    }
  }

  const toggleKnowledgeBaseChecked = (
    knowledgeBaseId: string,
    event: MouseEvent<HTMLButtonElement | HTMLInputElement>
  ) => {
    event.stopPropagation()
    setCheckedKnowledgeBaseIds((prev) =>
      prev.includes(knowledgeBaseId)
        ? prev.filter((item) => item !== knowledgeBaseId)
        : [...prev, knowledgeBaseId]
    )
  }

  const toggleDocumentChecked = (
    documentId: string,
    event: MouseEvent<HTMLButtonElement | HTMLInputElement>
  ) => {
    event.stopPropagation()
    setCheckedDocumentIds((prev) =>
      prev.includes(documentId)
        ? prev.filter((item) => item !== documentId)
        : [...prev, documentId]
    )
  }

  const knowledgeBasePageTotal = getPageTotal(
    knowledgeBaseTotal,
    KNOWLEDGE_BASE_PAGE_SIZE
  )
  const documentPageTotal = getPageTotal(documentTotal, DOCUMENT_PAGE_SIZE)
  const chatDisabled = useKnowledgeBase && !selectedKnowledgeBase
  const chatModeLabel = useKnowledgeBase ? '知识库 RAG' : '通用 Agent'
  const visibleChunkTotal = knowledgeBases.reduce((sum, item) => sum + item.chunk_count, 0)
  const documentChunkPageTotal = selectedDocumentDetail
    ? getPageTotal(selectedDocumentDetail.total_chunks, selectedDocumentDetail.page_size)
    : 1
  const managementPageTitleMap: Record<KnowledgePage, string> = {
    libraries: '知识库',
    'library-detail': '知识库详情',
    'document-detail': '知识详情',
    users: '用户管理',
  }
  const managementPageDescriptionMap: Record<KnowledgePage, string> = {
    libraries: '先从知识库卡片中选择目标知识库，再进入它的详情页和知识明细。',
    'library-detail': '为当前知识库上传知识文件，管理已有知识，并进入知识详情页查看切片。',
    'document-detail': '查看当前知识文档的切片明细、原始信息和删除操作。',
    users: '单独管理用户身份，切换后会隔离知识库和对话数据。',
  }
  const renderKnowledgePage = () => {
    if (knowledgePage === 'users') {
      return (
        <div className={styles.managementPageGrid}>
          <section className={styles.managementCard}>
            <div className={styles.managementHeader}>
              <h3>当前用户</h3>
              <span className={styles.managementMeta}>前端本地管理</span>
            </div>
            <div className={styles.managementMetaPanel}>
              <span>当前用户: {userId}</span>
              <span>知识库数量: {knowledgeBaseTotal}</span>
              <span>最近切片总数: {visibleChunkTotal}</span>
            </div>
            <div className={styles.managementForm}>
              <input
                className={styles.managementInput}
                value={userIdDraft}
                onChange={(event) => setUserIdDraft(event.target.value)}
                placeholder="输入或创建用户 ID"
              />
              <div className={styles.managementToolbar}>
                <button className={styles.managementButton} onClick={applyUserId}>
                  切换到该用户
                </button>
                <button
                  className={styles.managementMinorButton}
                  onClick={() => persistSavedUsers([userIdDraft, ...savedUsers])}
                >
                  保存到用户列表
                </button>
              </div>
            </div>
          </section>

          <section className={styles.managementCard}>
            <div className={styles.managementHeader}>
              <h3>用户列表</h3>
              <span className={styles.managementMeta}>{savedUsers.length} 个</span>
            </div>
            <div className={styles.managementList}>
              {savedUsers.length === 0 ? (
                <div className={styles.managementEmpty}>暂无保存的用户</div>
              ) : (
                savedUsers.map((savedUser) => (
                  <div
                    key={savedUser}
                    className={`${styles.managementListItemStatic} ${
                      savedUser === userId ? styles.managementListItemActive : ''
                    }`}
                  >
                    <div className={styles.managementListHeader}>
                      <strong>{savedUser}</strong>
                      <span>{savedUser === userId ? '当前用户' : '可切换'}</span>
                    </div>
                    <div className={styles.managementActionRow}>
                      <button
                        className={styles.managementMinorButton}
                        onClick={() => switchUser(savedUser)}
                      >
                        使用该用户
                      </button>
                      <button
                        className={styles.managementDangerMinorButton}
                        onClick={() => removeSavedUser(savedUser)}
                        disabled={savedUser === userId && savedUsers.length <= 1}
                      >
                        移除
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      )
    }

    if (knowledgePage === 'document-detail') {
      return selectedDocumentDetail ? (
        <div className={styles.managementPageGrid}>
          <section className={styles.managementCard}>
            <div className={styles.managementHeader}>
              <h3>{selectedDocumentDetail.document.display_name}</h3>
              <span className={styles.managementMeta}>
                {loadingDocumentDetail ? '加载中...' : `${selectedDocumentDetail.total_chunks} 个切片`}
              </span>
            </div>
            <div className={styles.managementMetaPanel}>
              <span>所属知识库: {selectedDocumentDetail.knowledge_base.name}</span>
              <span>原始文件: {selectedDocumentDetail.document.file_name}</span>
              <span>文件大小: {Math.max(1, Math.round(selectedDocumentDetail.document.file_size / 1024))} KB</span>
              <span>切片数量: {selectedDocumentDetail.document.chunk_count}</span>
              <span>更新时间: {formatDateTime(selectedDocumentDetail.document.updated_at)}</span>
            </div>
            <div className={styles.managementToolbar}>
              <button
                className={styles.managementMinorButton}
                onClick={() => navigateTo('knowledge', 'library-detail')}
              >
                返回知识库详情
              </button>
              <button
                className={styles.managementMinorButton}
                onClick={() => void renameDocument(selectedDocumentDetail.document)}
              >
                重命名
              </button>
              <button
                className={styles.managementDangerButton}
                onClick={() =>
                  void deleteDocument(
                    selectedDocumentDetail.document.document_id,
                    selectedDocumentDetail.document.display_name
                  )
                }
              >
                删除该知识
              </button>
            </div>
          </section>

          <section className={styles.managementCard}>
            <div className={styles.managementHeader}>
              <h3>切片详情</h3>
              <span className={styles.managementMeta}>
                第 {documentChunkPage} / {documentChunkPageTotal} 页
              </span>
            </div>
            <div className={styles.managementList}>
              {selectedDocumentDetail.chunks.length === 0 ? (
                <div className={styles.managementEmpty}>暂无切片数据</div>
              ) : (
                selectedDocumentDetail.chunks.map((chunk) => (
                  <div key={chunk.chunk_id} className={styles.managementListItemStatic}>
                    <div className={styles.managementListHeader}>
                      <strong>切片 #{chunk.segment_id || '-'}</strong>
                      <span>{chunk.chunk_id}</span>
                    </div>
                    <p className={styles.managementDescription}>{chunk.content}</p>
                    <div className={styles.managementListMeta}>
                      <span>页码: {String(chunk.metadata.pages_number ?? '-')}</span>
                      <span>标题: {String(chunk.metadata.title ?? '-')}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
            <Pagination
              page={documentChunkPage}
              pageTotal={documentChunkPageTotal}
              total={selectedDocumentDetail.total_chunks}
              onPrev={() => setDocumentChunkPage((prev) => Math.max(1, prev - 1))}
              onNext={() =>
                setDocumentChunkPage((prev) => Math.min(documentChunkPageTotal, prev + 1))
              }
            />
          </section>
        </div>
      ) : (
        <div className={styles.managementEmptyState}>
          <div className={styles.managementEmpty}>请先从知识库详情页选择一条知识</div>
          <button
            className={styles.managementButton}
            onClick={() => navigateTo('knowledge', 'libraries')}
          >
            返回知识库
          </button>
        </div>
      )
    }

    if (knowledgePage === 'library-detail') {
      return selectedKnowledgeBase ? (
        <div className={styles.managementWorkspace}>
          <section className={styles.managementHero}>
            <div className={styles.managementHeroCopy}>
              <span className={styles.managementHeroEyebrow}>Knowledge Base</span>
              <h2>{selectedKnowledgeBase.name}</h2>
              <p>{selectedKnowledgeBase.description || '当前知识库暂无描述。'}</p>
            </div>
            <div className={styles.managementHeroActions}>
              <button
                className={styles.managementMinorButton}
                onClick={() => navigateTo('knowledge', 'libraries')}
              >
                返回知识库列表
              </button>
              <button
                className={styles.managementDangerButton}
                onClick={() => void deleteKnowledgeBase()}
              >
                删除知识库
              </button>
            </div>
          </section>

          <div className={styles.managementSummaryGrid}>
            <div className={styles.managementSummaryCard}>
              <span className={styles.managementSummaryLabel}>文档总数</span>
              <strong className={styles.managementSummaryValue}>
                {selectedKnowledgeBase.document_count}
              </strong>
              <span className={styles.managementMeta}>当前知识库知识文档数量</span>
            </div>
            <div className={styles.managementSummaryCard}>
              <span className={styles.managementSummaryLabel}>切片总数</span>
              <strong className={styles.managementSummaryValue}>
                {selectedKnowledgeBase.chunk_count}
              </strong>
              <span className={styles.managementMeta}>切片会写入 ES 图索引</span>
            </div>
            <div className={styles.managementSummaryCard}>
              <span className={styles.managementSummaryLabel}>图前缀</span>
              <strong className={styles.managementSummaryValue}>
                {selectedKnowledgeBase.index_prefix}
              </strong>
              <span className={styles.managementMeta}>当前图检索命名空间</span>
            </div>
            <div className={styles.managementSummaryCard}>
              <span className={styles.managementSummaryLabel}>更新时间</span>
              <strong className={styles.managementSummaryValue}>
                {formatDateTime(selectedKnowledgeBase.updated_at)}
              </strong>
              <span className={styles.managementMeta}>最近一次知识库更新时间</span>
            </div>
          </div>

          <div className={styles.managementPageGrid}>
            <section className={styles.managementCard}>
              <div className={styles.managementHeader}>
                <h3>知识库设置</h3>
                <span className={styles.managementMeta}>名称与描述</span>
              </div>
              <div className={styles.managementForm}>
                <input
                  className={styles.managementInput}
                  value={selectedKnowledgeBaseName}
                  onChange={(event) => setSelectedKnowledgeBaseName(event.target.value)}
                  placeholder="知识库名称"
                />
                <input
                  className={styles.managementInput}
                  value={selectedKnowledgeBaseDescription}
                  onChange={(event) => setSelectedKnowledgeBaseDescription(event.target.value)}
                  placeholder="知识库描述"
                />
                <div className={styles.managementToolbar}>
                  <button
                    className={styles.managementButton}
                    disabled={savingKnowledgeBase}
                    onClick={() => void saveKnowledgeBase()}
                  >
                    保存设置
                  </button>
                </div>
              </div>

              <div className={styles.managementDivider} />

              <div className={styles.managementHeader}>
                <h3>添加知识</h3>
                <span className={styles.managementMeta}>支持 PDF / DOCX 等文件</span>
              </div>
              <div className={styles.managementToolbar}>
                <button
                  className={styles.managementButton}
                  disabled={uploadingDocuments}
                  onClick={openUploadDialog}
                >
                  {uploadingDocuments ? '上传中...' : '上传知识文件'}
                </button>
                <input
                  ref={uploadInputRef}
                  className={styles.hiddenUpload}
                  type="file"
                  multiple
                  onChange={handleUploadFiles}
                />
              </div>

              <div className={styles.managementMetaPanel}>
                <span>Passage: {selectedKnowledgeBase.passage_index}</span>
                <span>Entity: {selectedKnowledgeBase.entity_index}</span>
                <span>Relation: {selectedKnowledgeBase.relation_index}</span>
              </div>
            </section>

            <section className={styles.managementCard}>
              <div className={styles.managementHeader}>
                <h3>知识列表</h3>
                <span className={styles.managementMeta}>
                  {loadingDocuments ? '加载中...' : `共 ${documentTotal} 条`}
                </span>
              </div>
              <div className={styles.managementToolbar}>
                <div className={styles.managementSearchGroup}>
                  <input
                    className={styles.managementInput}
                    value={documentSearchInput}
                    onChange={(event) => setDocumentSearchInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        setDocumentPage(1)
                        setDocumentSearch(documentSearchInput.trim())
                      }
                    }}
                    placeholder="搜索知识文件"
                  />
                  <button
                    className={styles.managementButton}
                    onClick={() => {
                      setDocumentPage(1)
                      setDocumentSearch(documentSearchInput.trim())
                    }}
                  >
                    搜索
                  </button>
                </div>
                <button
                  className={styles.managementDangerButton}
                  disabled={checkedDocumentIds.length === 0 || deletingBulk}
                  onClick={() => void bulkDeleteDocuments()}
                >
                  批量删除
                </button>
              </div>

              <div className={styles.managementCardGrid}>
                {documents.length === 0 ? (
                  <div className={styles.managementEmpty}>当前知识库还没有知识文档</div>
                ) : (
                  documents.map((document) => (
                    <div key={document.document_id} className={styles.managementTileCard}>
                      <div className={styles.managementListHeader}>
                        <label
                          className={styles.managementCheckbox}
                          onClick={(event) =>
                            toggleDocumentChecked(
                              document.document_id,
                              event as unknown as MouseEvent<
                                HTMLButtonElement | HTMLInputElement
                              >
                            )
                          }
                        >
                          <input
                            type="checkbox"
                            checked={checkedDocumentIds.includes(document.document_id)}
                            onChange={() => undefined}
                          />
                        </label>
                        <strong>{document.display_name}</strong>
                        <span>{document.chunk_count} 切片</span>
                      </div>
                      <p className={styles.managementDescription}>
                        原始文件: {document.file_name}
                      </p>
                      <div className={styles.managementListMeta}>
                        <span>{Math.max(1, Math.round(document.file_size / 1024))} KB</span>
                        <span>{formatDateTime(document.updated_at)}</span>
                      </div>
                      <div className={styles.managementActionRow}>
                        <button
                          className={styles.managementButton}
                          onClick={() => openDocumentDetail(document)}
                        >
                          查看详情
                        </button>
                        <button
                          className={styles.managementMinorButton}
                          onClick={() => void renameDocument(document)}
                        >
                          重命名
                        </button>
                        <button
                          className={styles.managementDangerMinorButton}
                          onClick={() =>
                            void deleteDocument(document.document_id, document.display_name)
                          }
                        >
                          删除
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <Pagination
                page={documentPage}
                pageTotal={documentPageTotal}
                total={documentTotal}
                onPrev={() => setDocumentPage((prev) => Math.max(1, prev - 1))}
                onNext={() =>
                  setDocumentPage((prev) => Math.min(documentPageTotal, prev + 1))
                }
              />
            </section>
          </div>
        </div>
      ) : (
        <div className={styles.managementEmptyState}>
          <div className={styles.managementEmpty}>请先从知识库列表选择一个知识库</div>
          <button
            className={styles.managementButton}
            onClick={() => navigateTo('knowledge', 'libraries')}
          >
            返回知识库列表
          </button>
        </div>
      )
    }

    return (
      <div className={styles.managementWorkspace}>
        <section className={styles.managementHero}>
          <div className={styles.managementHeroCopy}>
            <span className={styles.managementHeroEyebrow}>Knowledge Bases</span>
            <h2>按知识库管理你的知识</h2>
            <p>这里展示当前用户下的所有知识库。点击卡片进入详情页，在详情页中继续添加知识文件和查看具体切片。</p>
          </div>
        </section>

        <div className={styles.managementToolbar}>
          <div className={styles.managementSearchGroup}>
            <input
              className={styles.managementInput}
              value={knowledgeBaseSearchInput}
              onChange={(event) => setKnowledgeBaseSearchInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  setKnowledgeBasePage(1)
                  setKnowledgeBaseSearch(knowledgeBaseSearchInput.trim())
                }
              }}
              placeholder="搜索知识库名称或描述"
            />
            <button
              className={styles.managementButton}
              onClick={() => {
                setKnowledgeBasePage(1)
                setKnowledgeBaseSearch(knowledgeBaseSearchInput.trim())
              }}
            >
              搜索
            </button>
          </div>
          <button
            className={styles.managementDangerButton}
            disabled={checkedKnowledgeBaseIds.length === 0 || deletingBulk}
            onClick={() => void bulkDeleteKnowledgeBases()}
          >
            批量删除
          </button>
        </div>

        <div className={styles.managementLibraryGrid}>
          <button
            type="button"
            className={styles.managementCreateCard}
            onClick={() => setShowCreateKnowledgeBaseModal(true)}
          >
            <span className={styles.managementCreateIcon}>+</span>
            <strong>新建知识库</strong>
            <span>点击后填写知识库名称和描述</span>
          </button>

          {knowledgeBases.length === 0 ? (
            <div className={styles.managementEmpty}>暂无知识库</div>
          ) : (
            knowledgeBases.map((knowledgeBase) => (
              <div
                key={knowledgeBase.knowledge_base_id}
                className={`${styles.managementLibraryCard} ${
                  selectedKnowledgeBaseId === knowledgeBase.knowledge_base_id
                    ? styles.managementListItemActive
                    : ''
                }`}
              >
                <div className={styles.managementListHeader}>
                  <label
                    className={styles.managementCheckbox}
                    onClick={(event) =>
                      toggleKnowledgeBaseChecked(
                        knowledgeBase.knowledge_base_id,
                        event as unknown as MouseEvent<
                          HTMLButtonElement | HTMLInputElement
                        >
                      )
                    }
                  >
                    <input
                      type="checkbox"
                      checked={checkedKnowledgeBaseIds.includes(
                        knowledgeBase.knowledge_base_id
                      )}
                      onChange={() => undefined}
                    />
                  </label>
                  <strong>{knowledgeBase.name}</strong>
                  <span>{knowledgeBase.document_count} 文档</span>
                </div>
                <p className={styles.managementDescription}>
                  {knowledgeBase.description || '暂无描述'}
                </p>
                <div className={styles.managementListMeta}>
                  <span>{knowledgeBase.chunk_count} 切片</span>
                  <span>{formatDateTime(knowledgeBase.updated_at)}</span>
                </div>
                <div className={styles.managementActionRow}>
                  <button
                    className={styles.managementButton}
                    onClick={() => openKnowledgeBaseLibrary(knowledgeBase)}
                  >
                    进入知识库
                  </button>
                  <button
                    className={styles.managementDangerMinorButton}
                    onClick={() =>
                      void deleteKnowledgeBase(knowledgeBase.knowledge_base_id)
                    }
                  >
                    删除
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <Pagination
          page={knowledgeBasePage}
          pageTotal={knowledgeBasePageTotal}
          total={knowledgeBaseTotal}
          onPrev={() => setKnowledgeBasePage((prev) => Math.max(1, prev - 1))}
          onNext={() =>
            setKnowledgeBasePage((prev) => Math.min(knowledgeBasePageTotal, prev + 1))
          }
        />
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.backgroundGrid} />
      <div className={styles.backgroundGlow} />

      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.logoArea}>
            <span className={styles.logoIcon}>AI</span>
            <h1 className={styles.title}>AI Agent Chat</h1>
          </div>
          <p className={styles.subtitle}>智能问答 · 知识库管理 · 图检索 RAG</p>
        </div>
      </header>

      <div className={styles.workspaceLayout}>
        <aside className={styles.sidebarNav}>
          <div className={styles.sidebarPanel}>
            <button
              className={`${styles.sidebarButton} ${
                viewMode === 'chat' ? styles.sidebarButtonActive : ''
              }`}
              onClick={() => navigateTo('chat')}
            >
              聊天
            </button>
            <button
              className={`${styles.sidebarButton} ${
                viewMode === 'knowledge' && knowledgePage !== 'users'
                  ? styles.sidebarButtonActive
                  : ''
              }`}
              onClick={() => navigateTo('knowledge', 'libraries')}
            >
              知识库
            </button>
            <button
              className={`${styles.sidebarButton} ${
                viewMode === 'knowledge' && knowledgePage === 'users'
                  ? styles.sidebarButtonActive
                  : ''
              }`}
              onClick={() => navigateTo('knowledge', 'users')}
            >
              用户管理
            </button>
          </div>
        </aside>

        <main className={styles.mainContent}>
          {viewMode === 'chat' ? (
            <>
              <div className={styles.sessionBar}>
                <div className={styles.sessionInfo}>
                  <span className={styles.sessionLabel}>会话</span>
                  <code className={styles.sessionId}>{sessionId.slice(0, 8)}...</code>
                  <span className={styles.sessionDivider}>|</span>
                  <span className={styles.sessionLabel}>用户</span>
                  <code className={styles.sessionId}>{userId}</code>
                  <span className={styles.sessionDivider}>|</span>
                  <span className={styles.sessionLabel}>模式</span>
                  <code className={styles.sessionId}>{chatModeLabel}</code>
                  <span className={styles.sessionDivider}>|</span>
                  <span className={styles.sessionLabel}>知识库</span>
                  <code className={styles.sessionId}>
                    {useKnowledgeBase
                      ? selectedKnowledgeBase?.name || '未选择'
                      : '未启用'}
                  </code>
                </div>
                <div className={styles.statusArea}>
                  <span className={`${styles.statusDot} ${styles[status]}`} />
                  <span className={styles.statusText}>
                    {status === 'ready' ? '就绪' : status === 'connecting' ? '处理中' : '错误'}
                  </span>
                  <button className={styles.clearBtn} onClick={clearChat}>
                    清空
                  </button>
                </div>
              </div>

              {showInterrupt && interruptData && (
                <div className={styles.interruptPanel}>
                  <div className={styles.interruptHeader}>
                    <span>需要人工确认</span>
                  </div>
                  <div className={styles.interruptContent}>
                    {interruptData.action_requests?.map((action, index) => (
                      <div key={index} className={styles.interruptAction}>
                        <div className={styles.interruptActionHeader}>
                          <span className={styles.interruptToolIcon}>
                            {getToolIcon(action.name)}
                          </span>
                          <span className={styles.interruptToolName}>{action.name}</span>
                        </div>
                        {action.description && (
                          <p className={styles.interruptDescription}>{action.description}</p>
                        )}
                        <div className={styles.interruptArgsSection}>
                          <label className={styles.interruptSectionLabel}>参数:</label>
                          <textarea
                            className={styles.interruptArgsEditor}
                            defaultValue={JSON.stringify(action.args, null, 2)}
                            rows={4}
                            id={`argsEditor${index}`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className={styles.interruptButtons}>
                    <button
                      className={`${styles.interruptBtn} ${styles.approve}`}
                      onClick={() => void handleInterruptAction('approve')}
                    >
                      批准
                    </button>
                    <button
                      className={`${styles.interruptBtn} ${styles.edit}`}
                      onClick={() => void handleInterruptAction('edit')}
                    >
                      编辑
                    </button>
                    <button
                      className={`${styles.interruptBtn} ${styles.reject}`}
                      onClick={() => void handleInterruptAction('reject')}
                    >
                      拒绝
                    </button>
                  </div>
                </div>
              )}

              <div className={styles.chatContainer} ref={chatContainerRef}>
                {messages.length === 0 ? (
                  <div className={styles.welcome}>
                    <div className={styles.welcomeIcon}>KB</div>
                    <h2>欢迎使用 AI Agent Chat</h2>
                    <p>
                      默认直接使用 Agent 对话。开启知识库后，将切到 RAG 图检索，并且必须选择一个知识库。
                    </p>
                    <div className={styles.exampleQueries}>
                      <button
                        onClick={() => setInputValue('请帮我总结一下今天要做的事情。')}
                      >
                        通用问答
                      </button>
                      <button
                        onClick={() =>
                          setInputValue('请列出文档中涉及的重要实体和它们之间的关系。')
                        }
                      >
                        提取实体关系
                      </button>
                      <button onClick={() => navigateTo('knowledge', 'libraries')}>
                        进入知识管理
                      </button>
                    </div>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <div key={msg.id} className={`${styles.message} ${styles[msg.role]}`}>
                      {msg.role === 'user' ? (
                        <>
                          <div className={styles.messageHeader}>
                            <div className={styles.avatar}>U</div>
                            <span className={styles.author}>用户</span>
                            <span className={styles.time}>
                              {new Date().toLocaleTimeString('zh-CN', {
                                hour: '2-digit',
                                minute: '2-digit',
                              })}
                            </span>
                          </div>
                          <div
                            className={styles.messageContent}
                            dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
                          />
                        </>
                      ) : (
                        <>
                          <div className={styles.messageHeader}>
                            <div className={`${styles.avatar} ${styles.ai}`}>AI</div>
                            <span className={styles.author}>AI 助手</span>
                            <span className={styles.time}>
                              {new Date().toLocaleTimeString('zh-CN', {
                                hour: '2-digit',
                                minute: '2-digit',
                              })}
                            </span>
                          </div>
                          {msg.reasoningContent && (
                            <div className={styles.reasoningContainer}>
                              <div className={styles.reasoningHeader}>
                                <span>深度思考</span>
                              </div>
                              <div
                                className={styles.reasoningContent}
                                dangerouslySetInnerHTML={{
                                  __html: parseMarkdown(msg.reasoningContent),
                                }}
                              />
                            </div>
                          )}
                          {msg.toolData?.map((toolData) => (
                            <ToolCard key={toolData.toolCall.id} toolData={toolData} />
                          ))}
                          {msg.content && (
                            <div
                              className={styles.messageContent}
                              dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
                            />
                          )}
                          {isProcessing &&
                            msg.id === currentAssistantMessageIdRef.current &&
                            !msg.content &&
                            !msg.reasoningContent &&
                            !msg.toolData?.length && (
                              <div className={styles.typingIndicator}>
                                <span />
                                <span />
                                <span />
                              </div>
                            )}
                        </>
                      )}
                    </div>
                  ))
                )}

                {isProcessing && messages[messages.length - 1]?.role !== 'ai' && (
                  <div className={`${styles.message} ${styles.ai}`}>
                    <div className={styles.messageHeader}>
                      <div className={`${styles.avatar} ${styles.ai}`}>AI</div>
                      <span className={styles.author}>AI 助手</span>
                    </div>
                    <div className={styles.typingIndicator}>
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                )}
              </div>

              <div className={styles.inputArea}>
                <div className={styles.inputContainer}>
                  <textarea
                    ref={textareaRef}
                    className={styles.input}
                    value={inputValue}
                    onChange={(event) => setInputValue(event.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                      useKnowledgeBase
                        ? chatDisabled
                          ? '请先在知识管理中选择一个知识库...'
                          : '输入您的知识库问题...'
                        : '输入您的问题...'
                    }
                    rows={1}
                  />
                  <div className={styles.toggles}>
                    <label className={styles.toggle}>
                      <input
                        type="checkbox"
                        checked={useKnowledgeBase}
                        onChange={(event) =>
                          handleKnowledgeBaseToggle(event.target.checked)
                        }
                      />
                      <span className={styles.toggleSlider} />
                      <span className={styles.toggleLabel}>知识库</span>
                    </label>
                    <label className={styles.toggle}>
                      <input
                        type="checkbox"
                        checked={internetSearch}
                        onChange={(event) => setInternetSearch(event.target.checked)}
                      />
                      <span className={styles.toggleSlider} />
                      <span className={styles.toggleLabel}>联网</span>
                    </label>
                    <label className={styles.toggle}>
                      <input
                        type="checkbox"
                        checked={deepThinking}
                        onChange={(event) => setDeepThinking(event.target.checked)}
                      />
                      <span className={styles.toggleSlider} />
                      <span className={styles.toggleLabel}>思考</span>
                    </label>
                  </div>
                  {useKnowledgeBase && (
                    <div className={styles.chatKnowledgeRow}>
                      <span className={styles.chatKnowledgeStatus}>
                        {selectedKnowledgeBase
                          ? `当前知识库：${selectedKnowledgeBase.name}`
                          : '已开启知识库问答，请先选择知识库'}
                      </span>
                      <button
                        className={styles.chatKnowledgeAction}
                        onClick={() => navigateTo('knowledge', 'libraries')}
                      >
                        {selectedKnowledgeBase ? '切换知识库' : '选择知识库'}
                      </button>
                    </div>
                  )}
                </div>
                {isProcessing ? (
                  <button className={styles.abortBtn} onClick={abortRequest}>
                    中断
                  </button>
                ) : (
                  <button
                    className={styles.sendBtn}
                    onClick={() => void sendMessage()}
                    disabled={!inputValue.trim() || chatDisabled}
                  >
                    发送
                  </button>
                )}
              </div>
            </>
          ) : (
            <div className={styles.managementPage}>
              <div className={styles.managementNoticeRow}>
                {managementNotice && (
                  <div className={styles.managementNotice}>{managementNotice}</div>
                )}
                {managementError && (
                  <div className={styles.managementError}>{managementError}</div>
                )}
              </div>
              <div className={styles.managementTopbar}>
                <div className={styles.managementRouteInfo}>
                  <span className={styles.managementBreadcrumb}>
                    知识管理 / {managementPageTitleMap[knowledgePage]}
                  </span>
                  <h2>{managementPageTitleMap[knowledgePage]}</h2>
                  <p>{managementPageDescriptionMap[knowledgePage]}</p>
                </div>
              </div>

              {renderKnowledgePage()}
            </div>
          )}
        </main>
      </div>

      {showCreateKnowledgeBaseModal && (
        <div
          className={styles.managementModalOverlay}
          onClick={() => setShowCreateKnowledgeBaseModal(false)}
        >
          <div
            className={styles.managementModal}
            onClick={(event) => event.stopPropagation()}
          >
            <div className={styles.managementHeader}>
              <h3>新建知识库</h3>
              <button
                type="button"
                className={styles.managementModalClose}
                onClick={() => setShowCreateKnowledgeBaseModal(false)}
              >
                关闭
              </button>
            </div>
            <div className={styles.managementForm}>
              <input
                className={styles.managementInput}
                value={knowledgeBaseName}
                onChange={(event) => setKnowledgeBaseName(event.target.value)}
                placeholder="知识库名称"
                autoFocus
              />
              <input
                className={styles.managementInput}
                value={knowledgeBaseDescription}
                onChange={(event) => setKnowledgeBaseDescription(event.target.value)}
                placeholder="知识库描述"
              />
              <div className={styles.managementToolbar}>
                <button
                  className={styles.managementButton}
                  disabled={savingKnowledgeBase}
                  onClick={() => void createKnowledgeBase()}
                >
                  创建知识库
                </button>
                <button
                  type="button"
                  className={styles.managementMinorButton}
                  onClick={() => setShowCreateKnowledgeBaseModal(false)}
                >
                  取消
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Pagination({
  page,
  pageTotal,
  total,
  onPrev,
  onNext,
}: {
  page: number
  pageTotal: number
  total: number
  onPrev: () => void
  onNext: () => void
}) {
  return (
    <div className={styles.managementPagination}>
      <button className={styles.managementMinorButton} onClick={onPrev} disabled={page <= 1}>
        上一页
      </button>
      <span className={styles.managementPaginationInfo}>
        第 {page} / {pageTotal} 页，共 {total} 条
      </span>
      <button
        className={styles.managementMinorButton}
        onClick={onNext}
        disabled={page >= pageTotal}
      >
        下一页
      </button>
    </div>
  )
}

function ToolCard({ toolData }: { toolData: ToolData }) {
  const [expanded, setExpanded] = useState(false)
  const toolName = toolData.toolCall?.name || 'tool'

  const argsStr = toolData.toolCall?.args
    ? JSON.stringify(toolData.toolCall.args, null, 2)
    : ''

  const firstArgKey = toolData.toolCall?.args
    ? Object.keys(toolData.toolCall.args)[0]
    : ''
  const firstArgValue =
    toolData.toolCall?.args && firstArgKey
      ? String(toolData.toolCall.args[firstArgKey]).slice(0, 30)
      : ''
  const argsSummary = firstArgKey
    ? `${firstArgKey}: ${firstArgValue}${firstArgValue.length >= 30 ? '...' : ''}`
    : ''

  return (
    <div
      className={`${styles.toolCard} ${expanded ? styles.expanded : ''}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className={styles.toolCardHeader}>
        <div className={styles.toolIcon}>{getToolIcon(toolName)}</div>
        <div className={styles.toolInfo}>
          <span className={styles.toolName}>{toolName}</span>
          <span className={styles.toolArgs}>{argsSummary}</span>
        </div>
        <div className={`${styles.toolStatus} ${styles.success}`}>OK</div>
        <div className={styles.toolExpandIcon}>{expanded ? '-' : '+'}</div>
      </div>
      {expanded && (
        <div className={styles.toolCardDetails}>
          {toolData.toolCall && (
            <div className={styles.toolDetailSection}>
              <span className={styles.toolDetailLabel}>输入参数</span>
              <pre className={`${styles.toolDetailCode} ${styles.inputCode}`}>{argsStr}</pre>
            </div>
          )}
          {toolData.toolOutput?.map((output, index) => (
            <div key={index} className={styles.toolDetailSection}>
              <span className={styles.toolDetailLabel}>执行结果</span>
              <pre className={`${styles.toolDetailCode} ${styles.outputCode}`}>
                {output.content}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
