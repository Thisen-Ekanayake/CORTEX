import type {
  Workspace,
  QueryLogEntry,
  CurrentQuery,
  User,
  DataSourceStatus,
} from '../types'

export const workspaces: Workspace[] = [
  { id: 'hr', label: 'HR' },
  { id: 'finance', label: 'Finance' },
  { id: 'legal', label: 'Legal' },
  { id: 'rd', label: 'R&D' },
]

export const queryLog: QueryLogEntry[] = [
  {
    id: '1',
    title: 'What changes were made in the 2024 travel reimbursement policy?',
    workspace: 'finance',
    mode: 'Analyze',
    timestamp: '2025-02-05T14:32:00Z',
    confidence: 0.92,
  },
  {
    id: '2',
    title: 'Compare Q3 and Q4 leave accrual rules',
    workspace: 'hr',
    mode: 'Compare',
    timestamp: '2025-02-05T13:15:00Z',
    confidence: 0.88,
  },
  {
    id: '3',
    title: 'Summarize contract amendments in vendor agreement',
    workspace: 'legal',
    mode: 'Summarize',
    timestamp: '2025-02-05T11:42:00Z',
    confidence: 0.95,
  },
]

export const currentQuery: CurrentQuery = {
  id: '1',
  queryText:
    'What changes were made in the 2024 travel reimbursement policy?',
  workspace: 'finance',
  mode: 'Analyze',
  timestamp: '2025-02-05T14:32:00Z',
  response: {
    executiveSummary:
      'The 2024 travel reimbursement policy introduces a higher per-diem cap for international travel, mandatory receipt submission within 30 days, and a new tier for economy-plus for flights over 6 hours. Mileage rate is unchanged.',
    detailedAnalysis:
      'Policy document v2.4 (effective Jan 2024) revises Section 4 (Travel) and Section 5 (Reimbursement). Key changes: (1) Per-diem for international travel increased from $85 to $120 per day with manager approval for high-cost cities. (2) All reimbursements require receipt upload to the portal within 30 days; late submissions require VP approval. (3) Flights over 6 hours may be booked economy-plus with prior approval. (4) Mileage rate remains $0.67/mile. (5) New exclusion for personal vehicle use when rental is available at destination.',
    keyFindings: [
      'International per-diem cap raised to $120/day (from $85).',
      '30-day receipt submission window is now mandatory; exceptions need VP approval.',
      'Economy-plus allowed for flights over 6 hours with approval.',
      'Mileage rate unchanged at $0.67/mile.',
      'Personal vehicle use disallowed when rental is available at destination.',
    ],
    riskNotes: [
      'Policy does not specify currency for international per-diem; assume USD.',
      'Economy-plus approval workflow is referenced but not detailed in this document.',
    ],
    confidenceScore: 0.92,
  },
  evidence: [
    {
      rank: 1,
      documentName: 'Travel_Reimbursement_Policy_2024.pdf',
      similarityScore: 0.94,
      pageNumber: 4,
      snippet:
        '...effective January 2024. Section 4.2: International per-diem shall not exceed $120 per day unless prior manager approval is obtained for high-cost cities as defined in Appendix B...',
    },
    {
      rank: 2,
      documentName: 'Travel_Reimbursement_Policy_2024.pdf',
      similarityScore: 0.89,
      pageNumber: 6,
      snippet:
        '...All expense reports must be submitted with supporting receipts within 30 calendar days of the expense date. Submissions after 30 days require written approval from the applicable Vice President...',
    },
    {
      rank: 3,
      documentName: 'Travel_Reimbursement_Policy_2024.pdf',
      similarityScore: 0.85,
      pageNumber: 5,
      snippet:
        '...For flights exceeding 6 hours, economy-plus or equivalent may be booked with prior approval from the department head. Mileage reimbursement rate remains $0.67 per mile...',
    },
  ],
}

export const user: User = {
  name: 'Alex Morgan',
  username: 'amorgan',
  role: 'Analyst',
}

export const dataSourceStatus: DataSourceStatus = {
  indexedCount: 1247,
  lastSyncTime: '2025-02-05T14:28:00Z',
  status: 'synced',
}
