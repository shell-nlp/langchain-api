'use client'

import { useState } from 'react'

import styles from '../ChatInterface.module.css'
import type { ReasoningBlock } from './types'
import { parseMarkdown } from './utils'

interface ReasoningCardProps {
  block: ReasoningBlock
}

export function ReasoningCard({ block }: ReasoningCardProps) {
  const [expanded, setExpanded] = useState(true)

  return (
    <section
      className={`${styles.reasoningContainer} ${expanded ? styles.expanded : ''}`}
    >
      <button
        type="button"
        className={styles.reasoningHeader}
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span>深度思考</span>
        <span className={styles.reasoningToggleIcon}>{expanded ? '-' : '+'}</span>
      </button>
      {expanded && (
        <div
          className={styles.reasoningContent}
          dangerouslySetInnerHTML={{
            __html: parseMarkdown(block.content),
          }}
        />
      )}
    </section>
  )
}
