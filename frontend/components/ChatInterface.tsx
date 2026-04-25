'use client'

import { useState, useRef, useEffect, useCallback, KeyboardEvent } from 'react'
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

const DEFAULT_BACKEND_URL = 'http://localhost:7869'

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

function generateSessionId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

function getToolIcon(toolName: string): string {
  const iconMap: Record<string, string> = {
    search: '🔍', calculator: '🧮', calc: '🧮', math: '📐',
    weather: '🌤️', time: '⏰', date: '📅', file: '📄',
    read: '📖', write: '✍️', edit: '✏️', api: '🌐',
    http: '🌐', request: '📡', fetch: '📡', python: '🐍',
    code: '💻', exec: '⚡', run: '▶️', bash: '💻',
    git: '📦', github: '🐙', email: '📧', translate: '🌐',
    analyze: '📊', chart: '📈', browser: '🌐'
  }
  const lower = toolName.toLowerCase()
  for (const [key, icon] of Object.entries(iconMap)) {
    if (lower.includes(key)) return icon
  }
  return '🔧'
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [status, setStatus] = useState<'ready' | 'connecting' | 'error'>('ready')
  const [isProcessing, setIsProcessing] = useState(false)
  const [internetSearch, setInternetSearch] = useState(false)
  const [deepThinking, setDeepThinking] = useState(false)
  const [showInterrupt, setShowInterrupt] = useState(false)
  const [interruptData, setInterruptData] = useState<StreamEvent['data']['__interrupt__'] | null>(null)
  
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentAssistantMessageIdRef = useRef<string | null>(null)
  const processedToolCallIdsRef = useRef<Set<string>>(new Set())

  const scrollToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const existingId = localStorage.getItem('chat_session_id')
      const nextId = existingId || generateSessionId()
      localStorage.setItem('chat_session_id', nextId)
      setSessionId(nextId)
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // 监听localStorage中的会话ID变化，确保多个标签页之间的同步
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const handleStorageChange = (event: StorageEvent) => {
        if (event.key === 'chat_session_id' && event.newValue) {
          setSessionId(event.newValue)
        }
      }
      
      window.addEventListener('storage', handleStorageChange)
      
      return () => {
        window.removeEventListener('storage', handleStorageChange)
      }
    }
  }, [])

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => {
      const existing = prev.find(m => m.id === message.id)
      if (existing) {
        return prev.map(m => m.id === message.id ? { ...m, ...message } : m)
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

  const updateAssistantMessage = useCallback((updater: (message: Message) => Message) => {
    const assistantMessageId = ensureAssistantMessage()
    setMessages(prev => prev.map(msg => (
      msg.id === assistantMessageId ? updater(msg) : msg
    )))
  }, [ensureAssistantMessage])

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

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    const data = event.data
    if (!data) return

    switch (event.event) {
      case 'token': {
        if (data.reasoning_token) {
          updateAssistantMessage(msg => ({
            ...msg,
            reasoningContent: `${msg.reasoningContent || ''}${data.reasoning_token}`
          }))
        }

        if (data.token) {
          updateAssistantMessage(msg => ({
            ...msg,
            content: `${msg.content}${data.token}`
          }))
        }
        break
      }

      case 'tool_calls': {
        if (data.tool_calls?.length) {
          updateAssistantMessage(msg => {
            const tools = [...(msg.toolData || [])]
            for (const toolCall of data.tool_calls || []) {
              if (!tools.some(tool => tool.toolCall.id === toolCall.id)) {
                tools.push({ toolCall, toolOutput: [] })
              }
            }
            return { ...msg, toolData: tools }
          })
        }
        break
      }

      case 'tool_output': {
        if (data.tool_output?.length) {
          updateAssistantMessage(msg => {
            let tools = [...(msg.toolData || [])]

            for (const output of data.tool_output || []) {
              if (processedToolCallIdsRef.current.has(output.tool_call_id)) {
                continue
              }
              processedToolCallIdsRef.current.add(output.tool_call_id)

              const normalizedOutput = {
                ...output,
                content: stringifyToolContent(output.content)
              }
              const existingToolIndex = tools.findIndex(tool => tool.toolCall.id === output.tool_call_id)

              if (existingToolIndex >= 0) {
                const existingTool = tools[existingToolIndex]
                tools = tools.map((tool, index) => (
                  index === existingToolIndex
                    ? {
                        ...existingTool,
                        toolOutput: [...(existingTool.toolOutput || []), normalizedOutput]
                      }
                    : tool
                ))
              } else {
                tools.push({
                  toolCall: {
                    id: output.tool_call_id,
                    name: 'tool',
                    args: {}
                  },
                  toolOutput: [normalizedOutput]
                })
              }
            }

            return { ...msg, toolData: tools }
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
  }, [updateAssistantMessage])

  const readEventStream = useCallback(async (response: Response) => {
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
          .map(line => line.trim())
          .filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trim())

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
  }, [handleStreamEvent])

  const sendMessage = async () => {
    const query = inputValue.trim()
    if (!query || isProcessing || !sessionId) return

    const userMsg: Message = {
      id: generateMessageId(),
      role: 'user',
      content: query
    }
    addMessage(userMsg)
    setInputValue('')
    
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    
    setIsProcessing(true)
    setStatus('connecting')

    const assistantMessageId = generateMessageId()
    currentAssistantMessageIdRef.current = assistantMessageId
    processedToolCallIdsRef.current.clear()
    addMessage({ id: assistantMessageId, role: 'ai', content: '', toolData: [] })

    abortControllerRef.current = new AbortController()

    const requestData = {
      query,
      session_id: sessionId,
      internet_search: internetSearch,
      deep_thinking: deepThinking
    }

    try {
      const response = await fetch(`${getApiBaseUrl()}/api/general_api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestData),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      await readEventStream(response)
    } catch (error: unknown) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Request failed:', error)
        addMessage({
          id: generateMessageId(),
          role: 'ai',
          content: `❌ 请求失败: ${error instanceof Error ? error.message : 'Unknown error'}`
        })
      }
    } finally {
      setIsProcessing(false)
      setStatus('ready')
      currentAssistantMessageIdRef.current = null
      processedToolCallIdsRef.current.clear()
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
    currentAssistantMessageIdRef.current = null
    processedToolCallIdsRef.current.clear()
    const newId = generateSessionId()
    localStorage.setItem('chat_session_id', newId)
    setSessionId(newId)
  }

  const abortRequest = () => {
    abortControllerRef.current?.abort()
  }

  const handleInterruptAction = async (decision: 'approve' | 'reject' | 'edit', editedActions?: Array<{ name: string; args: Record<string, unknown> }>) => {
    setShowInterrupt(false)
    
    const decisions = (interruptData?.action_requests || []).map((_, index) => {
      const obj: { type: string; edited_action?: { name: string; args: Record<string, unknown> }; message?: string } = { type: decision }
      if (decision === 'edit' && editedActions?.[index]) {
        obj.edited_action = editedActions[index]
      }
      return obj
    })

    addMessage({
      id: generateMessageId(),
      role: 'user',
      content: decision === 'approve' ? '✅ 已批准操作继续执行' 
        : decision === 'reject' ? '❌ 已拒绝操作' 
        : '✏️ 已编辑参数并继续执行'
    })

    setIsProcessing(true)
    setStatus('connecting')
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(`${getApiBaseUrl()}/api/general_api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ resume: { decisions }, session_id: sessionId }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      await readEventStream(response)
    } catch (error: unknown) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Request failed:', error)
        addMessage({
          id: generateMessageId(),
          role: 'ai',
          content: `❌ 请求失败: ${error instanceof Error ? error.message : 'Unknown error'}`
        })
      }
    } finally {
      setIsProcessing(false)
      setStatus('ready')
      setInterruptData(null)
      processedToolCallIdsRef.current.clear()
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.backgroundGrid} />
      <div className={styles.backgroundGlow} />
      
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.logoArea}>
            <span className={styles.logoIcon}>◈</span>
            <h1 className={styles.title}>AI Agent Chat</h1>
          </div>
          <p className={styles.subtitle}>智能助手 · 支持工具调用和流式响应</p>
        </div>
      </header>

      <div className={styles.sessionBar}>
        <div className={styles.sessionInfo}>
          <span className={styles.sessionLabel}>会话</span>
          <code className={styles.sessionId}>{sessionId.slice(0, 8)}...</code>
        </div>
        <div className={styles.statusArea}>
          <span className={`${styles.statusDot} ${styles[status]}`} />
          <span className={styles.statusText}>
            {status === 'ready' ? '就绪' : status === 'connecting' ? '处理中' : '错误'}
          </span>
          <button className={styles.clearBtn} onClick={clearChat}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
            </svg>
            清空
          </button>
        </div>
      </div>

      {showInterrupt && interruptData && (
        <div className={styles.interruptPanel}>
          <div className={styles.interruptHeader}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span>需要人工确认</span>
          </div>
          <div className={styles.interruptContent}>
            {interruptData.action_requests?.map((action, index) => (
              <div key={index} className={styles.interruptAction}>
                <div className={styles.interruptActionHeader}>
                  <span className={styles.interruptToolIcon}>{getToolIcon(action.name)}</span>
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
            <button className={`${styles.interruptBtn} ${styles.approve}`} onClick={() => handleInterruptAction('approve')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              批准
            </button>
            <button className={`${styles.interruptBtn} ${styles.edit}`} onClick={() => handleInterruptAction('edit')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
              编辑
            </button>
            <button className={`${styles.interruptBtn} ${styles.reject}`} onClick={() => handleInterruptAction('reject')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
              拒绝
            </button>
          </div>
        </div>
      )}

      <div className={styles.chatContainer} ref={chatContainerRef}>
        {messages.length === 0 ? (
          <div className={styles.welcome}>
            <div className={styles.welcomeIcon}>✦</div>
            <h2>欢迎使用 AI Agent Chat</h2>
            <p>我可以帮您执行各种任务，包括计算、查询、数据处理等。</p>
            <div className={styles.exampleQueries}>
              <button onClick={() => setInputValue('请你执行如下任务：\n1. 计算 10 + 10 的结果\n2. 将结果乘以 5\n3. 根据结果生成一个故事')}>
                🧮 数学计算示例
              </button>
              <button onClick={() => setInputValue('查询今天的天气情况')}>
                🌤️ 天气查询
              </button>
              <button onClick={() => setInputValue('帮我分析一下当前的市场趋势')}>
                📊 数据分析
              </button>
            </div>
          </div>
        ) : (
          messages.map(msg => (
            <div key={msg.id} className={`${styles.message} ${styles[msg.role]}`}>
              {msg.role === 'user' ? (
                <>
                  <div className={styles.messageHeader}>
                    <div className={styles.avatar}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                      </svg>
                    </div>
                    <span className={styles.author}>{"\u7528\u6237"}</span>
                    <span className={styles.time}>{new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  <div className={styles.messageContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
                </>
              ) : (
                <>
                  <div className={styles.messageHeader}>
                    <div className={`${styles.avatar} ${styles.ai}`}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2M7.5 13a1.5 1.5 0 100 3 1.5 1.5 0 000-3m9 0a1.5 1.5 0 100 3 1.5 1.5 0 000-3"/>
                      </svg>
                    </div>
                    <span className={styles.author}>{"AI \u52a9\u624b"}</span>
                    <span className={styles.time}>{new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  {msg.reasoningContent && (
                    <div className={styles.reasoningContainer}>
                      <div className={styles.reasoningHeader}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10"/>
                          <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>
                          <line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        <span>{"\u6df1\u5ea6\u601d\u8003"}</span>
                      </div>
                      <div className={styles.reasoningContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.reasoningContent) }} />
                    </div>
                  )}
                  {msg.toolData?.map(toolData => (
                    <ToolCard key={toolData.toolCall.id} toolData={toolData} />
                  ))}
                  {msg.content && (
                    <div className={styles.messageContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
                  )}
                  {isProcessing && msg.id === currentAssistantMessageIdRef.current && !msg.content && !msg.reasoningContent && !msg.toolData?.length && (
                    <div className={styles.typingIndicator}>
                      <span /><span /><span />
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
              <div className={`${styles.avatar} ${styles.ai}`}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2M7.5 13a1.5 1.5 0 100 3 1.5 1.5 0 000-3m9 0a1.5 1.5 0 100 3 1.5 1.5 0 000-3"/>
                </svg>
              </div>
              <span className={styles.author}>AI 助手</span>
            </div>
            <div className={styles.typingIndicator}>
              <span /><span /><span />
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
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入您的问题或任务..."
            rows={1}
          />
          <div className={styles.toggles}>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={internetSearch}
                onChange={e => setInternetSearch(e.target.checked)}
              />
              <span className={styles.toggleSlider} />
              <span className={styles.toggleLabel}>🌐 联网</span>
            </label>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={deepThinking}
                onChange={e => setDeepThinking(e.target.checked)}
              />
              <span className={styles.toggleSlider} />
              <span className={styles.toggleLabel}>🧠 思考</span>
            </label>
          </div>
        </div>
        {isProcessing ? (
          <button className={styles.abortBtn} onClick={abortRequest}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            中断
          </button>
        ) : (
          <button className={styles.sendBtn} onClick={sendMessage} disabled={!inputValue.trim()}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
            发送
          </button>
        )}
      </div>
    </div>
  )
}

function ToolCard({ toolData }: { toolData: ToolData }) {
  const [expanded, setExpanded] = useState(false)
  const toolName = toolData.toolCall?.name || 'tool'
  const toolIcon = getToolIcon(toolName)
  
  const argsStr = toolData.toolCall?.args 
    ? JSON.stringify(toolData.toolCall.args, null, 2)
    : ''
  
  const firstArgKey = toolData.toolCall?.args ? Object.keys(toolData.toolCall.args)[0] : ''
  const firstArgValue = toolData.toolCall?.args && firstArgKey 
    ? String(toolData.toolCall.args[firstArgKey]).slice(0, 30) 
    : ''
  const argsSummary = firstArgKey ? `${firstArgKey}: ${firstArgValue}${firstArgValue.length >= 30 ? '...' : ''}` : ''

  return (
    <div className={`${styles.toolCard} ${expanded ? styles.expanded : ''}`} onClick={() => setExpanded(!expanded)}>
      <div className={styles.toolCardHeader}>
        <div className={styles.toolIcon}>{toolIcon}</div>
        <div className={styles.toolInfo}>
          <span className={styles.toolName}>{toolName}</span>
          <span className={styles.toolArgs}>{argsSummary}</span>
        </div>
        <div className={`${styles.toolStatus} ${styles.success}`}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        </div>
        <div className={styles.toolExpandIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </div>
      </div>
      {expanded && (
        <div className={styles.toolCardDetails}>
          {toolData.toolCall && (
            <div className={styles.toolDetailSection}>
              <span className={styles.toolDetailLabel}>输入参数</span>
              <pre className={`${styles.toolDetailCode} ${styles.input}`}>{argsStr}</pre>
            </div>
          )}
          {toolData.toolOutput?.map((output, index) => (
            <div key={index} className={styles.toolDetailSection}>
              <span className={styles.toolDetailLabel}>执行结果</span>
              <pre className={`${styles.toolDetailCode} ${styles.output}`}>{output.content}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
