// NOTE: Overview stats, query-volume chart and activity feed are now served
// live from the cortex_api `/api/stats` endpoint (see pages/Home.tsx). The
// former hardcoded `stats`, `chartData` and `alerts` arrays were removed.

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
