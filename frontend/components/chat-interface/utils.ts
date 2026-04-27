'use client'

import { marked } from 'marked'

import {
  DEFAULT_BACKEND_URL,
  DEFAULT_KNOWLEDGE_PAGE,
} from './constants'
import type { KnowledgePage, ViewMode } from './types'

export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL
  if (envUrl) return envUrl.replace(/\/$/, '')

  if (typeof window === 'undefined') return ''

  const { hostname, origin } = window.location
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return DEFAULT_BACKEND_URL
  }

  return origin
}

export function getApiUrl(path: string): string {
  return `${getApiBaseUrl()}${path}`
}

export function normalizeKnowledgePage(value?: string): KnowledgePage {
  switch (value) {
    case 'library-detail':
    case 'document-detail':
    case 'users':
      return value
    default:
      return 'libraries'
  }
}

export function getRouteHash(
  viewMode: ViewMode,
  knowledgePage: KnowledgePage = DEFAULT_KNOWLEDGE_PAGE
): string {
  return viewMode === 'chat' ? '#/chat' : `#/knowledge/${knowledgePage}`
}

export function parseRouteHash(hash: string): {
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

export function generateSessionId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

export function formatDateTime(value: string): string {
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

export function getPageTotal(total: number, pageSize: number): number {
  return Math.max(1, Math.ceil(total / pageSize))
}

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
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

export function getToolIcon(toolName: string): string {
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

export function escapeHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

export function parseMarkdown(text: string): string {
  try {
    return marked.parse(text, { async: false }) as string
  } catch {
    return escapeHtml(text)
  }
}

export function stringifyToolContent(content: unknown): string {
  if (typeof content === 'string') return content
  if (content == null) return ''
  try {
    return JSON.stringify(content, null, 2)
  } catch {
    return String(content)
  }
}
