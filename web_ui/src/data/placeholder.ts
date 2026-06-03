export const stats = [
  { title: 'Documents indexed', value: '1,247', subtitle: 'Last sync: 2 min ago', trend: 'up' as const, trendValue: '+12' },
  { title: 'Queries today', value: '89', subtitle: 'RAG: 62 · Chat: 27', trend: 'up' as const, trendValue: '+8%' },
  { title: 'Router accuracy', value: '94%', subtitle: '7-day rolling', trend: 'neutral' as const, trendValue: '—' },
  { title: 'Active sessions', value: '3', subtitle: 'You + 2 background', trend: 'down' as const, trendValue: '-1' },
]

export const chartData = [
  { name: 'Mon', queries: 42, rag: 28 },
  { name: 'Tue', queries: 58, rag: 41 },
  { name: 'Wed', queries: 51, rag: 35 },
  { name: 'Thu', queries: 67, rag: 48 },
  { name: 'Fri', queries: 89, rag: 62 },
  { name: 'Sat', queries: 34, rag: 22 },
  { name: 'Sun', queries: 45, rag: 31 },
]

export const alerts = [
  { id: '1', type: 'info' as const, message: 'RAG index updated with 3 new documents.', time: '5m ago' },
  { id: '2', type: 'success' as const, message: 'Router confidence threshold tuned.', time: '1h ago' },
  { id: '3', type: 'warning' as const, message: 'High memory usage on last query.', time: '2h ago' },
]

export const searchResults = [
  { id: '1', title: 'CORTEX Architecture Overview', snippet: '...intelligent query routing to RAG, CHAT, or META handlers. Reinforcement learning improves routing over time.', score: 0.94, source: 'README.md' },
  { id: '2', title: 'Document Ingestion Pipeline', snippet: '...load PDF, TXT, and DOCX into Chroma. RecursiveCharacterTextSplitter with 1200 chars, 200 overlap.', score: 0.87, source: 'ingest.md' },
  { id: '3', title: 'RL Router Weights', snippet: '...TF-IDF classifier with confidence scores. Q-learning style updates from user feedback.', score: 0.82, source: 'rl_router.md' },
]

export const documents = [
  { id: '1', name: 'README.md', type: 'Markdown', size: '12 KB', updated: '2 hours ago', preview: 'CORTEX is a local, privacy-first RAG system with intelligent query routing...' },
  { id: '2', name: 'requirements.txt', type: 'Text', size: '1 KB', updated: '1 day ago', preview: 'langchain\nchromadb\ntransformers\ntorch\nrich\n...' },
  { id: '3', name: 'documentation/embeddings.md', type: 'Markdown', size: '8 KB', updated: '3 days ago', preview: 'Embeddings use all-MiniLM-L6-v2 (384 dimensions)...' },
  { id: '4', name: 'tutorial/5_question_answering.ipynb', type: 'Jupyter', size: '24 KB', updated: '1 week ago', preview: 'RAG chain implementation with document retrieval...' },
]

export const chatMessages = [
  { role: 'user' as const, content: 'What can CORTEX do?' },
  { role: 'assistant' as const, content: 'CORTEX is your local RAG assistant. It can:\n\n• **Answer questions** from your ingested documents (PDF, TXT, DOCX)\n• **Route queries** intelligently—RAG for document questions, CHAT for conversation, META for system questions\n• **Learn from feedback** via reinforcement learning to improve routing\n• **Stream responses** in real time\n\nYou can ask me anything about your knowledge base or have a general conversation.' },
]
