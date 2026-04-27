export interface Message {
  id: string
  role: 'user' | 'ai'
  content: string
  reasoningContent?: string
  toolData?: ToolData[]
}

export interface ToolData {
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

export interface StreamEvent {
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

export interface KnowledgeBase {
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

export interface KnowledgeDocument {
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

export interface UploadResult {
  knowledge_base: KnowledgeBase
  documents: KnowledgeDocument[]
  errors: Array<{
    file_name: string
    error: string
  }>
}

export interface PaginatedKnowledgeBaseResponse {
  items: KnowledgeBase[]
  total: number
  page: number
  page_size: number
}

export interface PaginatedKnowledgeDocumentResponse {
  items: KnowledgeDocument[]
  total: number
  page: number
  page_size: number
}

export interface BulkDeleteKnowledgeBaseResponse {
  deleted_ids: string[]
  failed: Record<string, string>
}

export interface BulkDeleteDocumentResponse {
  deleted_ids: string[]
  failed: Record<string, string>
  knowledge_base?: KnowledgeBase | null
}

export type ViewMode = 'chat' | 'knowledge'
export type KnowledgePage =
  | 'libraries'
  | 'library-detail'
  | 'document-detail'
  | 'users'
export type RequestMode = 'agent' | 'rag'
export type ChatStatus = 'ready' | 'connecting' | 'error'
export type InterruptData = StreamEvent['data']['__interrupt__']

export interface KnowledgeChunk {
  chunk_id: string
  document_id: string
  segment_id?: number | string | null
  content: string
  metadata: Record<string, unknown>
}

export interface KnowledgeDocumentDetailResponse {
  knowledge_base: KnowledgeBase
  document: KnowledgeDocument
  chunks: KnowledgeChunk[]
  total_chunks: number
  page: number
  page_size: number
}
