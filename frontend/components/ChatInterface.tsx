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

import styles from './ChatInterface.module.css'
import { ChatView } from './chat-interface/ChatView'
import {
  DEFAULT_AGENT_API_PATH,
  DEFAULT_KNOWLEDGE_PAGE,
  DEFAULT_RAG_API_PATH,
  DOCUMENT_CHUNK_PAGE_SIZE,
  DOCUMENT_PAGE_SIZE,
  KB_BULK_DELETE_API_PATH,
  KB_CREATE_API_PATH,
  KB_DELETE_API_PATH,
  KB_DETAIL_API_PATH,
  KB_DOCUMENT_BULK_DELETE_API_PATH,
  KB_DOCUMENT_DELETE_API_PATH,
  KB_DOCUMENT_DETAIL_API_PATH,
  KB_DOCUMENT_LIST_API_PATH,
  KB_DOCUMENT_UPDATE_API_PATH,
  KB_DOCUMENT_UPLOAD_API_PATH,
  KB_LIST_API_PATH,
  KB_UPDATE_API_PATH,
  KNOWLEDGE_BASE_PAGE_SIZE,
} from './chat-interface/constants'
import { KnowledgeManagementView } from './chat-interface/KnowledgeManagementView'
import type {
  AssistantMessageItem,
  BulkDeleteDocumentResponse,
  BulkDeleteKnowledgeBaseResponse,
  InterruptData,
  KnowledgeBase,
  KnowledgeDocument,
  KnowledgeDocumentDetailResponse,
  KnowledgePage,
  Message,
  PaginatedKnowledgeBaseResponse,
  PaginatedKnowledgeDocumentResponse,
  RequestMode,
  ReasoningBlock,
  StreamEvent,
  UploadResult,
  ViewMode,
} from './chat-interface/types'
import {
  fetchJson,
  generateMessageId,
  generateSessionId,
  getApiUrl,
  getPageTotal,
  getRouteHash,
  parseRouteHash,
  stringifyToolContent,
} from './chat-interface/utils'

function appendReasoningToken(
  blocks: ReasoningBlock[] | undefined,
  items: AssistantMessageItem[] | undefined,
  token: string,
  startNewBlock: boolean,
  createBlockId: () => string,
  legacyContent?: string
): { reasoningBlocks: ReasoningBlock[]; messageItems: AssistantMessageItem[] } {
  const normalizedBlocks =
    blocks && blocks.length > 0
      ? blocks
      : legacyContent
        ? [{ id: createBlockId(), content: legacyContent }]
        : []
  const normalizedItems =
    items && items.length > 0
      ? items
      : normalizedBlocks.map((block) => ({
          id: `reasoning_item_${block.id}`,
          type: 'reasoning' as const,
          reasoningBlockId: block.id,
        }))

  if (startNewBlock || normalizedBlocks.length === 0) {
    const id = createBlockId()
    return {
      reasoningBlocks: [...normalizedBlocks, { id, content: token }],
      messageItems: [
        ...normalizedItems,
        {
          id: `reasoning_item_${id}`,
          type: 'reasoning',
          reasoningBlockId: id,
        },
      ],
    }
  }

  return {
    reasoningBlocks: normalizedBlocks.map((block, index) =>
      index === normalizedBlocks.length - 1
        ? { ...block, content: `${block.content}${token}` }
        : block
    ),
    messageItems: normalizedItems,
  }
}

function appendContentToken(
  blocks: ReasoningBlock[] | undefined,
  items: AssistantMessageItem[] | undefined,
  token: string,
  appendToLastBlock: boolean,
  createBlockId: () => string
): { contentBlocks: ReasoningBlock[]; messageItems: AssistantMessageItem[] } {
  const normalizedBlocks = blocks || []
  const normalizedItems = items || []

  if (appendToLastBlock && normalizedBlocks.length > 0) {
    return {
      contentBlocks: normalizedBlocks.map((block, index) =>
        index === normalizedBlocks.length - 1
          ? { ...block, content: `${block.content}${token}` }
          : block
      ),
      messageItems: normalizedItems,
    }
  }

  const id = createBlockId()
  return {
    contentBlocks: [...normalizedBlocks, { id, content: token }],
    messageItems: [
      ...normalizedItems,
      {
        id: `content_item_${id}`,
        type: 'content',
        contentBlockId: id,
      },
    ],
  }
}

function ensureToolItem(
  items: AssistantMessageItem[] | undefined,
  toolCallId: string
): AssistantMessageItem[] {
  const normalizedItems = items || []
  if (
    normalizedItems.some(
      (item) => item.type === 'tool' && item.toolCallId === toolCallId
    )
  ) {
    return normalizedItems
  }

  return [
    ...normalizedItems,
    {
      id: `tool_item_${toolCallId}`,
      type: 'tool',
      toolCallId,
    },
  ]
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
  const [interruptData, setInterruptData] = useState<InterruptData | null>(null)

  const [userId, setUserId] = useState('demo-user')
  const [userIdDraft, setUserIdDraft] = useState('demo-user')

  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [knowledgeBaseTotal, setKnowledgeBaseTotal] = useState(0)
  const [knowledgeBasePage, setKnowledgeBasePage] = useState(1)
  const [knowledgeBaseSearchInput, setKnowledgeBaseSearchInput] = useState('')
  const [knowledgeBaseSearch, setKnowledgeBaseSearch] = useState('')

  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState('')
  const [selectedKnowledgeBase, setSelectedKnowledgeBase] =
    useState<KnowledgeBase | null>(null)
  const [selectedKnowledgeBaseName, setSelectedKnowledgeBaseName] = useState('')
  const [selectedKnowledgeBaseDescription, setSelectedKnowledgeBaseDescription] =
    useState('')
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
  const [showCreateKnowledgeBaseModal, setShowCreateKnowledgeBaseModal] =
    useState(false)
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
  const lastAssistantStreamEventRef = useRef<
    'reasoning' | 'content' | 'tool' | 'interrupt' | null
  >(null)
  const reasoningBlockCounterRef = useRef(0)
  const contentBlockCounterRef = useRef(0)
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
    lastAssistantStreamEventRef.current = null
    reasoningBlockCounterRef.current = 0
    contentBlockCounterRef.current = 0
    const newId = generateSessionId()
    localStorage.setItem('rag_chat_session_id', newId)
    setSessionId(newId)
  }, [])

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => {
      const existing = prev.find((item) => item.id === message.id)
      if (existing) {
        return prev.map((item) =>
          item.id === message.id ? { ...item, ...message } : item
        )
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

  const loadKnowledgeBaseDetail = useCallback(
    async (targetUserId: string, knowledgeBaseId: string) => {
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
    },
    []
  )

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
              page_size: DOCUMENT_CHUNK_PAGE_SIZE,
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
    const normalized = Array.from(
      new Set(nextUsers.map((item) => item.trim()).filter(Boolean))
    )
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
      setManagementNotice(`Deleted ${result.deleted_ids.length} knowledge base(s).`)
      if (Object.keys(result.failed).length > 0) {
        setManagementError(
          Object.entries(result.failed)
            .map(([id, message]) => `${id}: ${message}`)
            .join('\n')
        )
      }
    } catch (error) {
      setManagementError(
        error instanceof Error
          ? error.message
          : 'Failed to bulk delete knowledge bases.'
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
      const result = await fetchJson<UploadResult>(getApiUrl(KB_DOCUMENT_UPLOAD_API_PATH), {
        method: 'POST',
        body: formData,
      })

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
            const shouldStartNewBlock =
              lastAssistantStreamEventRef.current !== 'reasoning'
            updateAssistantMessage((message) => {
              const { reasoningBlocks, messageItems } = appendReasoningToken(
                message.reasoningBlocks,
                message.messageItems,
                data.reasoning_token || '',
                shouldStartNewBlock,
                () => {
                  reasoningBlockCounterRef.current += 1
                  return `${message.id}_reasoning_${reasoningBlockCounterRef.current}`
                },
                message.reasoningContent
              )

              return {
                ...message,
                reasoningBlocks,
                messageItems,
                reasoningContent: `${message.reasoningContent || ''}${data.reasoning_token}`,
              }
            })
            lastAssistantStreamEventRef.current = 'reasoning'
          }

          if (data.token) {
            const shouldAppendToLastBlock =
              lastAssistantStreamEventRef.current === 'content'
            updateAssistantMessage((message) => {
              const { contentBlocks, messageItems } = appendContentToken(
                message.contentBlocks,
                message.messageItems,
                data.token || '',
                shouldAppendToLastBlock,
                () => {
                  contentBlockCounterRef.current += 1
                  return `${message.id}_content_${contentBlockCounterRef.current}`
                }
              )

              return {
                ...message,
                content: `${message.content}${data.token}`,
                contentBlocks,
                messageItems,
              }
            })
            lastAssistantStreamEventRef.current = 'content'
          }
          break
        }

        case 'tool_calls': {
          if (data.tool_calls?.length) {
            updateAssistantMessage((message) => {
              const tools = [...(message.toolData || [])]
              let messageItems = message.messageItems || []
              for (const toolCall of data.tool_calls || []) {
                if (!tools.some((tool) => tool.toolCall.id === toolCall.id)) {
                  tools.push({ toolCall, toolOutput: [] })
                }
                messageItems = ensureToolItem(messageItems, toolCall.id)
              }
              return { ...message, toolData: tools, messageItems }
            })
            lastAssistantStreamEventRef.current = 'tool'
          }
          break
        }

        case 'tool_output': {
          if (data.tool_output?.length) {
            updateAssistantMessage((message) => {
              let tools = [...(message.toolData || [])]
              let messageItems = message.messageItems || []

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
                messageItems = ensureToolItem(messageItems, output.tool_call_id)
              }

              return { ...message, toolData: tools, messageItems }
            })
            lastAssistantStreamEventRef.current = 'tool'
          }
          break
        }

        case '__interrupt__': {
          if (data.__interrupt__) {
            setInterruptData(data.__interrupt__)
            setShowInterrupt(true)
            setIsProcessing(false)
            setStatus('ready')
            lastAssistantStreamEventRef.current = 'interrupt'
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
    lastAssistantStreamEventRef.current = null
    reasoningBlockCounterRef.current = 0
    contentBlockCounterRef.current = 0
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
        getApiUrl(requestMode === 'rag' ? DEFAULT_RAG_API_PATH : DEFAULT_AGENT_API_PATH),
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
      lastAssistantStreamEventRef.current = null
      reasoningBlockCounterRef.current = 0
      contentBlockCounterRef.current = 0
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
      setManagementError(
        '当前中断来自知识库问答，但未找到对应知识库，请重新发起知识库对话。'
      )
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
    lastAssistantStreamEventRef.current = null
    reasoningBlockCounterRef.current = 0
    contentBlockCounterRef.current = 0
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
        getApiUrl(requestMode === 'rag' ? DEFAULT_RAG_API_PATH : DEFAULT_AGENT_API_PATH),
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
      currentAssistantMessageIdRef.current = null
      processedToolCallIdsRef.current.clear()
      lastAssistantStreamEventRef.current = null
      reasoningBlockCounterRef.current = 0
      contentBlockCounterRef.current = 0
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
            <ChatView
              messages={messages}
              sessionId={sessionId}
              userId={userId}
              chatModeLabel={chatModeLabel}
              status={status}
              isProcessing={isProcessing}
              useKnowledgeBase={useKnowledgeBase}
              selectedKnowledgeBaseName={selectedKnowledgeBase?.name || null}
              showInterrupt={showInterrupt}
              interruptData={interruptData}
              inputValue={inputValue}
              chatDisabled={chatDisabled}
              internetSearch={internetSearch}
              deepThinking={deepThinking}
              currentAssistantMessageId={currentAssistantMessageIdRef.current}
              chatContainerRef={chatContainerRef}
              textareaRef={textareaRef}
              onClearChat={clearChat}
              onInterruptAction={handleInterruptAction}
              onInputChange={setInputValue}
              onKeyDown={handleKeyDown}
              onKnowledgeBaseToggle={handleKnowledgeBaseToggle}
              onInternetSearchChange={setInternetSearch}
              onDeepThinkingChange={setDeepThinking}
              onNavigateToKnowledge={() => navigateTo('knowledge', 'libraries')}
              onAbortRequest={abortRequest}
              onSendMessage={sendMessage}
            />
          ) : (
            <KnowledgeManagementView
              knowledgePage={knowledgePage}
              managementNotice={managementNotice}
              managementError={managementError}
              userId={userId}
              userIdDraft={userIdDraft}
              savedUsers={savedUsers}
              knowledgeBaseTotal={knowledgeBaseTotal}
              visibleChunkTotal={visibleChunkTotal}
              knowledgeBases={knowledgeBases}
              selectedKnowledgeBaseId={selectedKnowledgeBaseId}
              selectedKnowledgeBase={selectedKnowledgeBase}
              selectedKnowledgeBaseName={selectedKnowledgeBaseName}
              selectedKnowledgeBaseDescription={selectedKnowledgeBaseDescription}
              checkedKnowledgeBaseIds={checkedKnowledgeBaseIds}
              knowledgeBaseSearchInput={knowledgeBaseSearchInput}
              knowledgeBasePage={knowledgeBasePage}
              knowledgeBasePageTotal={knowledgeBasePageTotal}
              documents={documents}
              documentTotal={documentTotal}
              documentPage={documentPage}
              documentPageTotal={documentPageTotal}
              documentSearchInput={documentSearchInput}
              checkedDocumentIds={checkedDocumentIds}
              selectedDocumentDetail={selectedDocumentDetail}
              documentChunkPage={documentChunkPage}
              documentChunkPageTotal={documentChunkPageTotal}
              knowledgeBaseName={knowledgeBaseName}
              knowledgeBaseDescription={knowledgeBaseDescription}
              showCreateKnowledgeBaseModal={showCreateKnowledgeBaseModal}
              savingKnowledgeBase={savingKnowledgeBase}
              uploadingDocuments={uploadingDocuments}
              deletingBulk={deletingBulk}
              loadingDocuments={loadingDocuments}
              loadingDocumentDetail={loadingDocumentDetail}
              uploadInputRef={uploadInputRef}
              onNavigateTo={navigateTo}
              onUserIdDraftChange={setUserIdDraft}
              onApplyUserId={applyUserId}
              onPersistSavedUsers={persistSavedUsers}
              onSwitchUser={switchUser}
              onRemoveSavedUser={removeSavedUser}
              onSelectedKnowledgeBaseNameChange={setSelectedKnowledgeBaseName}
              onSelectedKnowledgeBaseDescriptionChange={
                setSelectedKnowledgeBaseDescription
              }
              onSaveKnowledgeBase={saveKnowledgeBase}
              onDeleteKnowledgeBase={deleteKnowledgeBase}
              onOpenUploadDialog={openUploadDialog}
              onHandleUploadFiles={handleUploadFiles}
              onDocumentSearchInputChange={setDocumentSearchInput}
              onDocumentPageChange={setDocumentPage}
              onDocumentSearchChange={setDocumentSearch}
              onBulkDeleteDocuments={bulkDeleteDocuments}
              onToggleDocumentChecked={toggleDocumentChecked}
              onOpenDocumentDetail={openDocumentDetail}
              onRenameDocument={renameDocument}
              onDeleteDocument={deleteDocument}
              onDocumentChunkPageChange={setDocumentChunkPage}
              onKnowledgeBaseSearchInputChange={setKnowledgeBaseSearchInput}
              onKnowledgeBasePageChange={setKnowledgeBasePage}
              onKnowledgeBaseSearchChange={setKnowledgeBaseSearch}
              onBulkDeleteKnowledgeBases={bulkDeleteKnowledgeBases}
              onShowCreateKnowledgeBaseModalChange={setShowCreateKnowledgeBaseModal}
              onToggleKnowledgeBaseChecked={toggleKnowledgeBaseChecked}
              onOpenKnowledgeBaseLibrary={openKnowledgeBaseLibrary}
              onKnowledgeBaseNameChange={setKnowledgeBaseName}
              onKnowledgeBaseDescriptionChange={setKnowledgeBaseDescription}
              onCreateKnowledgeBase={createKnowledgeBase}
            />
          )}
        </main>
      </div>
    </div>
  )
}
