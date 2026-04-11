'use client'

import { useState, useRef, useEffect, useCallback, KeyboardEvent } from 'react'
import { marked } from 'marked'
import styles from './ChatInterface.module.css'

interface Message {
  id: string
  role: 'user' | 'ai'
  content: string
  reasoningContent?: string
  isToolCall?: boolean
  toolData?: ToolData
}

interface ToolData {
  toolCall?: {
    name: string
    args: Record<string, unknown>
  }
  toolOutput?: Array<{
    tool_call_id: string
    content: string
  }>
}

interface StreamEvent {
  event: string
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
      content: string
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

const API_BASE_URL = typeof window !== 'undefined' 
  ? (window.location.origin.includes('localhost') ? 'http://localhost:7869' : window.location.origin)
  : ''

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
  const [sessionId] = useState(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('chat_session_id')
      if (stored) return stored
      const newId = generateSessionId()
      localStorage.setItem('chat_session_id', newId)
      return newId
    }
    return generateSessionId()
  })
  const [status, setStatus] = useState<'ready' | 'connecting' | 'error'>('ready')
  const [isProcessing, setIsProcessing] = useState(false)
  const [internetSearch, setInternetSearch] = useState(false)
  const [deepThinking, setDeepThinking] = useState(false)
  const [showInterrupt, setShowInterrupt] = useState(false)
  const [interruptData, setInterruptData] = useState<StreamEvent['data']['__interrupt__'] | null>(null)
  
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const messageIdMapRef = useRef<Map<string, string>>(new Map())
  const contentAccumulatorRef = useRef<Map<string, string>>(new Map())
  const reasoningAccumulatorRef = useRef<Map<string, string>>(new Map())
  const pendingToolCallsRef = useRef<Map<string, { tool_calls: StreamEvent['data']['tool_calls'], messageId: string }>>(new Map())

  const scrollToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => {
      const existing = prev.find(m => m.id === message.id)
      if (existing) {
        return prev.map(m => m.id === message.id ? { ...m, ...message } : m)
      }
      return [...prev, message]
    })
  }, [])

  const updateMessageContent = useCallback((messageId: string, content: string, isReasoning = false) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId) {
        if (isReasoning) {
          return { ...msg, reasoningContent: content }
        }
        return { ...msg, content }
      }
      return msg
    }))
  }, [])

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

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    const data = event.data
    if (!data) return

    switch (event.event) {
      case 'token': {
        const messageId = data.id || ''
        
        if (data.reasoning_token) {
          const current = reasoningAccumulatorRef.current.get(messageId) || ''
          const updated = current + data.reasoning_token
          reasoningAccumulatorRef.current.set(messageId, updated)
          
          if (messageIdMapRef.current.has(messageId)) {
            updateMessageContent(messageId, updated, true)
          } else {
            const newMsg: Message = {
              id: messageId,
              role: 'ai',
              content: '',
              reasoningContent: data.reasoning_token
            }
            addMessage(newMsg)
            messageIdMapRef.current.set(messageId, messageId)
          }
        }
        
        if (data.token) {
          const current = contentAccumulatorRef.current.get(messageId) || ''
          const updated = current + data.token
          contentAccumulatorRef.current.set(messageId, updated)
          
          if (messageIdMapRef.current.has(messageId)) {
            updateMessageContent(messageId, updated, false)
          } else {
            const newMsg: Message = {
              id: messageId,
              role: 'ai',
              content: data.token
            }
            addMessage(newMsg)
            messageIdMapRef.current.set(messageId, messageId)
          }
        }
        break
      }
      
      case 'tool_calls': {
        if (data.tool_calls && data.id) {
          const msgId = data.id
          if (messageIdMapRef.current.has(msgId)) {
            messageIdMapRef.current.delete(msgId)
            contentAccumulatorRef.current.delete(msgId)
            setMessages(prev => prev.filter(m => m.id !== msgId))
          }
          pendingToolCallsRef.current.set(msgId, {
            tool_calls: data.tool_calls,
            messageId: msgId
          })
        }
        break
      }
      
      case 'tool_output': {
        if (data.tool_output) {
          let matchedCall: { id: string; name: string; args: Record<string, unknown> } | null = null
          let matchedMsgId: string | null = null
          
          for (const [msgId, pendingData] of pendingToolCallsRef.current) {
            const match = pendingData.tool_calls?.find(call =>
              data.tool_output?.some(output => output.tool_call_id === call.id)
            )
            if (match) {
              matchedCall = match
              matchedMsgId = msgId
              break
            }
          }
          
          if (matchedCall && matchedMsgId) {
            pendingToolCallsRef.current.delete(matchedMsgId)
            
            const toolMsg: Message = {
              id: matchedMsgId,
              role: 'ai',
              content: '',
              isToolCall: true,
              toolData: {
                toolCall: matchedCall,
                toolOutput: data.tool_output
              }
            }
            addMessage(toolMsg)
            messageIdMapRef.current.set(matchedMsgId, matchedMsgId)
          } else {
            const toolMsg: Message = {
              id: generateMessageId(),
              role: 'ai',
              content: '',
              isToolCall: true,
              toolData: { toolOutput: data.tool_output }
            }
            addMessage(toolMsg)
          }
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
  }, [addMessage, updateMessageContent])

  const sendMessage = async () => {
    const query = inputValue.trim()
    if (!query || isProcessing) return

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
    
    abortControllerRef.current = new AbortController()

    const requestData = {
      query,
      session_id: sessionId,
      internet_search: internetSearch,
      deep_thinking: deepThinking
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/general_api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestData),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim().startsWith('data:')) {
            const jsonStr = line.trim().slice(5).trim()
            try {
              const event = JSON.parse(jsonStr) as StreamEvent
              handleStreamEvent(event)
            } catch (e) {
              console.error('Parse error:', e)
            }
          }
        }
      }
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
      messageIdMapRef.current.clear()
      contentAccumulatorRef.current.clear()
      reasoningAccumulatorRef.current.clear()
      pendingToolCallsRef.current.clear()
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
    const newId = generateSessionId()
    localStorage.setItem('chat_session_id', newId)
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
      const response = await fetch(`${API_BASE_URL}/api/general_api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ resume: { decisions }, session_id: sessionId }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim().startsWith('data:')) {
            const jsonStr = line.trim().slice(5).trim()
            try {
              const event = JSON.parse(jsonStr) as StreamEvent
              handleStreamEvent(event)
            } catch (e) {
              console.error('Parse error:', e)
            }
          }
        }
      }
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
      messageIdMapRef.current.clear()
      contentAccumulatorRef.current.clear()
      reasoningAccumulatorRef.current.clear()
      pendingToolCallsRef.current.clear()
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
                    <span className={styles.author}>用户</span>
                    <span className={styles.time}>{new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  <div className={styles.messageContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
                </>
              ) : (
                <>
                  {msg.isToolCall && msg.toolData ? (
                    <ToolCard toolData={msg.toolData} />
                  ) : (
                    <>
                      <div className={styles.messageHeader}>
                        <div className={`${styles.avatar} ${styles.ai}`}>
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2M7.5 13a1.5 1.5 0 100 3 1.5 1.5 0 000-3m9 0a1.5 1.5 0 100 3 1.5 1.5 0 000-3"/>
                          </svg>
                        </div>
                        <span className={styles.author}>AI 助手</span>
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
                            <span>深度思考</span>
                          </div>
                          <div className={styles.reasoningContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.reasoningContent) }} />
                        </div>
                      )}
                      {msg.content && (
                        <div className={styles.messageContent} dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
                      )}
                    </>
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
