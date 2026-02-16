export type WorkspaceId = 'hr' | 'finance' | 'legal' | 'rd'

export type IntelligenceMode =
  | 'Ask'
  | 'Analyze'
  | 'Compare'
  | 'Summarize'
  | 'Draft'
  | 'Investigate'

export type ModelId = 'cortex-fast' | 'cortex-pro' | 'cortex-deep'

export interface Workspace {
  id: WorkspaceId
  label: string
}

export interface QueryLogEntry {
  id: string
  title: string
  workspace: WorkspaceId
  mode: IntelligenceMode
  timestamp: string
  confidence: number
}

export interface EvidenceItem {
  rank: number
  documentName: string
  similarityScore: number
  pageNumber?: number
  snippet: string
}

export interface IntelligenceResponse {
  executiveSummary: string
  detailedAnalysis: string
  keyFindings: string[]
  riskNotes: string[]
  confidenceScore: number
}

export interface CurrentQuery {
  id: string
  queryText: string
  workspace: WorkspaceId
  mode: IntelligenceMode
  timestamp: string
  response: IntelligenceResponse
  evidence: EvidenceItem[]
}

export interface User {
  name: string
  username: string
  role: string
}

export interface DataSourceStatus {
  indexedCount: number
  lastSyncTime: string
  status: 'synced' | 'syncing' | 'pending'
}
