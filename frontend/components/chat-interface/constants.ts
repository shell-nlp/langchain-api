import type { KnowledgePage } from './types'

export const DEFAULT_BACKEND_URL = 'http://localhost:7869'
export const DEFAULT_AGENT_API_PATH = '/api/agent/general_api'
export const DEFAULT_RAG_API_PATH = '/api/rag/general_api'
export const KB_LIST_API_PATH = '/api/rag/knowledge-bases/list'
export const KB_CREATE_API_PATH = '/api/rag/knowledge-bases/create'
export const KB_DETAIL_API_PATH = '/api/rag/knowledge-bases/detail'
export const KB_UPDATE_API_PATH = '/api/rag/knowledge-bases/update'
export const KB_DELETE_API_PATH = '/api/rag/knowledge-bases/delete'
export const KB_BULK_DELETE_API_PATH = '/api/rag/knowledge-bases/bulk-delete'
export const KB_DOCUMENT_LIST_API_PATH = '/api/rag/knowledge-bases/documents/list'
export const KB_DOCUMENT_DETAIL_API_PATH = '/api/rag/knowledge-bases/documents/detail'
export const KB_DOCUMENT_UPLOAD_API_PATH = '/api/rag/knowledge-bases/documents/upload'
export const KB_DOCUMENT_UPDATE_API_PATH = '/api/rag/knowledge-bases/documents/update'
export const KB_DOCUMENT_DELETE_API_PATH = '/api/rag/knowledge-bases/documents/delete'
export const KB_DOCUMENT_BULK_DELETE_API_PATH =
  '/api/rag/knowledge-bases/documents/bulk-delete'

export const KNOWLEDGE_BASE_PAGE_SIZE = 8
export const DOCUMENT_PAGE_SIZE = 10
export const DOCUMENT_CHUNK_PAGE_SIZE = 8
export const DEFAULT_KNOWLEDGE_PAGE: KnowledgePage = 'libraries'

export const MANAGEMENT_PAGE_TITLE_MAP: Record<KnowledgePage, string> = {
  libraries: '知识库',
  'library-detail': '知识库详情',
  'document-detail': '知识详情',
  users: '用户管理',
}

export const MANAGEMENT_PAGE_DESCRIPTION_MAP: Record<KnowledgePage, string> = {
  libraries: '先从知识库卡片中选择目标知识库，再进入它的详情页和知识明细。',
  'library-detail':
    '为当前知识库上传知识文件，管理已有知识，并进入知识详情页查看切片。',
  'document-detail': '查看当前知识文档的切片明细、原始信息和删除操作。',
  users: '单独管理用户身份，切换后会隔离知识库和对话数据。',
}
