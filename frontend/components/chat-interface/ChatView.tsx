'use client'

import type { KeyboardEvent, Ref } from 'react'

import styles from '../ChatInterface.module.css'
import { ToolCard } from './ToolCard'
import type {
  ChatStatus,
  InterruptData,
  Message,
} from './types'
import { getToolIcon, parseMarkdown } from './utils'

interface ChatViewProps {
  messages: Message[]
  sessionId: string
  userId: string
  chatModeLabel: string
  status: ChatStatus
  isProcessing: boolean
  useKnowledgeBase: boolean
  selectedKnowledgeBaseName: string | null
  showInterrupt: boolean
  interruptData: InterruptData | null
  inputValue: string
  chatDisabled: boolean
  internetSearch: boolean
  deepThinking: boolean
  currentAssistantMessageId: string | null
  chatContainerRef: Ref<HTMLDivElement>
  textareaRef: Ref<HTMLTextAreaElement>
  onClearChat: () => void
  onInterruptAction: (decision: 'approve' | 'reject' | 'edit') => void | Promise<void>
  onInputChange: (value: string) => void
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  onKnowledgeBaseToggle: (checked: boolean) => void
  onInternetSearchChange: (checked: boolean) => void
  onDeepThinkingChange: (checked: boolean) => void
  onNavigateToKnowledge: () => void
  onAbortRequest: () => void
  onSendMessage: () => void | Promise<void>
}

export function ChatView({
  messages,
  sessionId,
  userId,
  chatModeLabel,
  status,
  isProcessing,
  useKnowledgeBase,
  selectedKnowledgeBaseName,
  showInterrupt,
  interruptData,
  inputValue,
  chatDisabled,
  internetSearch,
  deepThinking,
  currentAssistantMessageId,
  chatContainerRef,
  textareaRef,
  onClearChat,
  onInterruptAction,
  onInputChange,
  onKeyDown,
  onKnowledgeBaseToggle,
  onInternetSearchChange,
  onDeepThinkingChange,
  onNavigateToKnowledge,
  onAbortRequest,
  onSendMessage,
}: ChatViewProps) {
  return (
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
            {useKnowledgeBase ? selectedKnowledgeBaseName || '未选择' : '未启用'}
          </code>
        </div>
        <div className={styles.statusArea}>
          <span className={`${styles.statusDot} ${styles[status]}`} />
          <span className={styles.statusText}>
            {status === 'ready' ? '就绪' : status === 'connecting' ? '处理中' : '错误'}
          </span>
          <button className={styles.clearBtn} onClick={onClearChat}>
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
              onClick={() => void onInterruptAction('approve')}
            >
              批准
            </button>
            <button
              className={`${styles.interruptBtn} ${styles.edit}`}
              onClick={() => void onInterruptAction('edit')}
            >
              编辑
            </button>
            <button
              className={`${styles.interruptBtn} ${styles.reject}`}
              onClick={() => void onInterruptAction('reject')}
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
              <button onClick={() => onInputChange('请帮我总结一下今天要做的事情。')}>
                通用问答
              </button>
              <button
                onClick={() =>
                  onInputChange('请列出文档中涉及的重要实体和它们之间的关系。')
                }
              >
                提取实体关系
              </button>
              <button onClick={onNavigateToKnowledge}>进入知识管理</button>
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
                    msg.id === currentAssistantMessageId &&
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
            onChange={(event) => onInputChange(event.target.value)}
            onKeyDown={onKeyDown}
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
                onChange={(event) => onKnowledgeBaseToggle(event.target.checked)}
              />
              <span className={styles.toggleSlider} />
              <span className={styles.toggleLabel}>知识库</span>
            </label>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={internetSearch}
                onChange={(event) => onInternetSearchChange(event.target.checked)}
              />
              <span className={styles.toggleSlider} />
              <span className={styles.toggleLabel}>联网</span>
            </label>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={deepThinking}
                onChange={(event) => onDeepThinkingChange(event.target.checked)}
              />
              <span className={styles.toggleSlider} />
              <span className={styles.toggleLabel}>思考</span>
            </label>
          </div>
          {useKnowledgeBase && (
            <div className={styles.chatKnowledgeRow}>
              <span className={styles.chatKnowledgeStatus}>
                {selectedKnowledgeBaseName
                  ? `当前知识库：${selectedKnowledgeBaseName}`
                  : '已开启知识库问答，请先选择知识库'}
              </span>
              <button className={styles.chatKnowledgeAction} onClick={onNavigateToKnowledge}>
                {selectedKnowledgeBaseName ? '切换知识库' : '选择知识库'}
              </button>
            </div>
          )}
        </div>
        {isProcessing ? (
          <button className={styles.abortBtn} onClick={onAbortRequest}>
            中断
          </button>
        ) : (
          <button
            className={styles.sendBtn}
            onClick={() => void onSendMessage()}
            disabled={!inputValue.trim() || chatDisabled}
          >
            发送
          </button>
        )}
      </div>
    </>
  )
}
