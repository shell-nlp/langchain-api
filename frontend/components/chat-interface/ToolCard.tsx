'use client'

import { useState } from 'react'

import styles from '../ChatInterface.module.css'
import type { ToolData } from './types'
import { getToolIcon } from './utils'

interface ToolCardProps {
  toolData: ToolData
}

export function ToolCard({ toolData }: ToolCardProps) {
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
              <pre className={`${styles.toolDetailCode} ${styles.inputCode}`}>
                {argsStr}
              </pre>
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
