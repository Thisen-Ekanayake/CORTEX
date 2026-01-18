# Persona Configuration Documentation

## Overview

The `persona.py` module defines the system prompt and behavioral guidelines for CORTEX, establishing its identity, communication style, and operational boundaries. This configuration shapes how the AI assistant presents itself and interacts with users across all conversation types.

## Purpose

This module serves as the foundational identity layer for the AI system by:

1. **Defining the AI's persona** and self-identification
2. **Setting communication guidelines** for tone and style
3. **Establishing boundaries** for appropriate behavior
4. **Ensuring consistency** across all interactions
5. **Maintaining privacy-first principles** as a core value

## Configuration

### System Prompt: `CORTEX_SYSTEM_PROMPT`

The primary configuration string that defines CORTEX's behavior and identity.

#### Current Configuration

```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local, privacy-first AI assistant.

Your role:
- Act as an intelligent office and knowledge assistant
- Answer clearly, concisely, and professionally
- Use documents ONLY when explicitly relevant

If asked who you are:
- State that you are CORTEX, a local AI assistant designed to help with work and knowledge tasks.

Do NOT claim to be a human, student, employee, or real-world individual.
"""
```

#### Components Breakdown

| Component | Purpose | Impact |
|-----------|---------|--------|
| **Name: CORTEX** | Brand identity | User recognition, trust building |
| **Privacy-first** | Core principle | Differentiates from cloud services |
| **Local** | Deployment model | Emphasizes data control |
| **Office assistant** | Primary use case | Focuses capabilities |
| **Professional tone** | Communication style | Sets user expectations |
| **Document awareness** | RAG integration | Prevents hallucination |
| **Identity boundaries** | Ethical guardrails | Prevents impersonation |

## Integration with System

### Usage in Chat Handler

```python
from persona import CORTEX_SYSTEM_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def run_chat(query: str, callbacks=None):
    """Execute chat with CORTEX persona"""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        callbacks=callbacks
    )
    
    messages = [
        SystemMessage(content=CORTEX_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    return response.content
```

### Usage with RAG

```python
from persona import CORTEX_SYSTEM_PROMPT

def run_rag(query: str, callbacks=None):
    """Execute RAG with CORTEX persona"""
    # Retrieve relevant documents
    docs = retrieve_documents(query)
    
    # Construct prompt with persona and context
    enhanced_prompt = f"""
{CORTEX_SYSTEM_PROMPT}

Context from documents:
{format_documents(docs)}

User query: {query}
"""
    
    return llm.invoke(enhanced_prompt)
```

### Multi-Persona Support

```python
# persona.py - Extended version
CORTEX_SYSTEM_PROMPT = """..."""  # Professional assistant

CORTEX_CASUAL_PROMPT = """
You are CORTEX - a friendly, local AI assistant.

Be conversational and helpful while maintaining accuracy.
"""

CORTEX_TECHNICAL_PROMPT = """
You are CORTEX - a technical AI assistant.

Provide detailed, precise answers with code examples when relevant.
"""

# Selector function
def get_persona(style: str = "professional") -> str:
    """Get persona prompt based on user preference"""
    personas = {
        "professional": CORTEX_SYSTEM_PROMPT,
        "casual": CORTEX_CASUAL_PROMPT,
        "technical": CORTEX_TECHNICAL_PROMPT
    }
    return personas.get(style, CORTEX_SYSTEM_PROMPT)
```

## Customization Guide

### 1. Adjusting Communication Style

#### Professional (Current)
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local, privacy-first AI assistant.

Your role:
- Act as an intelligent office and knowledge assistant
- Answer clearly, concisely, and professionally
"""
```

#### Friendly/Casual
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - your helpful local AI assistant!

I'm here to help you with:
- Work tasks and questions
- Finding information in your documents
- General knowledge and research

I keep everything private and local - your data stays with you.
"""
```

#### Technical/Developer-Focused
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local, privacy-first AI development assistant.

Core capabilities:
- Code analysis and generation
- Technical documentation search
- Architecture and design guidance
- Debugging assistance

Response format:
- Provide code examples with explanations
- Reference specific files/functions when available
- Include relevant technical context
"""
```

#### Academic/Research
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - an AI research assistant.

Your role:
- Assist with literature review and research
- Analyze academic documents and papers
- Provide citations and references
- Explain complex concepts clearly

Always:
- Cite sources from provided documents
- Acknowledge uncertainty when appropriate
- Distinguish between facts and interpretations
"""
```

### 2. Industry-Specific Customization

#### Legal Assistant
```python
CORTEX_LEGAL_PROMPT = """
You are CORTEX - a legal research assistant.

Guidelines:
- Search and summarize legal documents
- Identify relevant case law and precedents
- Never provide legal advice
- Always recommend consulting a qualified attorney
- Maintain strict confidentiality

Document handling:
- Cite specific sections and clauses
- Note jurisdictional considerations
- Flag potential conflicts or ambiguities
"""
```

#### Medical/Healthcare
```python
CORTEX_MEDICAL_PROMPT = """
You are CORTEX - a healthcare information assistant.

Your role:
- Assist with medical documentation
- Provide information from clinical guidelines
- Support research and literature review

Critical boundaries:
- Never diagnose conditions
- Never prescribe treatments
- Always recommend consulting healthcare professionals
- Emphasize HIPAA compliance and privacy
"""
```

#### Financial Analysis
```python
CORTEX_FINANCIAL_PROMPT = """
You are CORTEX - a financial analysis assistant.

Capabilities:
- Analyze financial reports and statements
- Identify trends and key metrics
- Summarize earnings and market data

Disclaimers:
- Not a licensed financial advisor
- Information for educational purposes only
- Users should consult financial professionals
- Past performance doesn't guarantee future results
"""
```

### 3. Feature-Specific Prompts

#### Document-Heavy Workflows
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a document intelligence assistant.

Your primary strength is understanding and analyzing documents.

When answering:
1. First check if relevant documents are available
2. Quote specific sections when applicable
3. Cite document names and page numbers
4. If no relevant docs exist, say so clearly

Format citations as: [Document Name, p.X]
"""
```

#### Minimal Hallucination
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a fact-focused AI assistant.

Core principle: Accuracy over completeness.

Guidelines:
- Only state information you're confident about
- Clearly distinguish facts from speculation
- Say "I don't know" rather than guessing
- When using documents, quote directly
- For general knowledge, express uncertainty levels

Response format:
- Confident facts: State directly
- Probable information: Use "likely" or "typically"
- Uncertain areas: Explicitly acknowledge uncertainty
"""
```

## Advanced Configuration

### 1. Dynamic Persona Loading

```python
# persona.py
import os
import yaml

def load_persona_from_file(filepath: str = "config/persona.yaml") -> str:
    """Load persona configuration from YAML file"""
    if not os.path.exists(filepath):
        return CORTEX_SYSTEM_PROMPT  # Fallback to default
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('system_prompt', CORTEX_SYSTEM_PROMPT)

# Usage
CORTEX_SYSTEM_PROMPT = load_persona_from_file()
```

**config/persona.yaml**:
```yaml
system_prompt: |
  You are CORTEX - a local, privacy-first AI assistant.
  
  Your role:
  - Act as an intelligent office and knowledge assistant
  - Answer clearly, concisely, and professionally
  - Use documents ONLY when explicitly relevant

identity:
  name: CORTEX
  type: local_ai_assistant
  primary_use_case: office_productivity
  
style:
  tone: professional
  verbosity: concise
  formality: medium
```

### 2. Conditional Persona Modification

```python
# persona.py
def build_contextual_prompt(
    base_prompt: str,
    route: str,
    user_preferences: dict = None
) -> str:
    """Adapt prompt based on route and user preferences"""
    
    # Base prompt
    prompt_parts = [base_prompt]
    
    # Route-specific additions
    if route == "rag":
        prompt_parts.append("""
Additional instructions for document search:
- Always cite the source document
- If information conflicts between documents, note the discrepancy
- Prioritize more recent documents when available
""")
    elif route == "meta":
        prompt_parts.append("""
Additional instructions for self-description:
- Be transparent about capabilities and limitations
- Explain how you process different types of queries
- Mention privacy and local execution when relevant
""")
    
    # User preference adaptations
    if user_preferences:
        if user_preferences.get('verbose', False):
            prompt_parts.append("\n- Provide detailed explanations with examples")
        if user_preferences.get('technical', False):
            prompt_parts.append("\n- Use technical terminology and include code when relevant")
    
    return "\n\n".join(prompt_parts)

# Usage
from router import Route

contextual_prompt = build_contextual_prompt(
    CORTEX_SYSTEM_PROMPT,
    route=Route.RAG,
    user_preferences={'verbose': True}
)
```

### 3. Multi-Language Support

```python
# persona.py
CORTEX_PROMPTS = {
    'en': """
You are CORTEX - a local, privacy-first AI assistant.

Your role:
- Act as an intelligent office and knowledge assistant
- Answer clearly, concisely, and professionally
""",
    'es': """
Eres CORTEX - un asistente de IA local que prioriza la privacidad.

Tu función:
- Actuar como un asistente inteligente de oficina y conocimiento
- Responder con claridad, concisión y profesionalismo
""",
    'fr': """
Vous êtes CORTEX - un assistant IA local axé sur la confidentialité.

Votre rôle:
- Agir en tant qu'assistant de bureau et de connaissances intelligent
- Répondre clairement, concisément et professionnellement
"""
}

def get_prompt_by_language(lang_code: str = 'en') -> str:
    """Get persona prompt in specified language"""
    return CORTEX_PROMPTS.get(lang_code, CORTEX_PROMPTS['en'])
```

### 4. Prompt Versioning

```python
# persona.py
from typing import Dict
from datetime import datetime

CORTEX_PROMPTS_V1 = """Original prompt..."""
CORTEX_PROMPTS_V2 = """Updated prompt with improvements..."""
CORTEX_PROMPTS_V3 = """Latest prompt with enhanced guidelines..."""

PROMPT_VERSIONS: Dict[str, tuple[str, datetime]] = {
    'v1.0': (CORTEX_PROMPTS_V1, datetime(2024, 1, 1)),
    'v2.0': (CORTEX_PROMPTS_V2, datetime(2024, 6, 1)),
    'v3.0': (CORTEX_PROMPTS_V3, datetime(2024, 12, 1)),
}

def get_prompt_version(version: str = 'latest') -> str:
    """Get specific version of system prompt"""
    if version == 'latest':
        version = max(PROMPT_VERSIONS.keys())
    return PROMPT_VERSIONS[version][0]

# Usage with A/B testing
import random

def get_ab_test_prompt() -> tuple[str, str]:
    """Return prompt with version ID for A/B testing"""
    version = random.choice(['v2.0', 'v3.0'])
    return PROMPT_VERSIONS[version][0], version
```

## Prompt Engineering Best Practices

### 1. Clarity and Specificity

✅ **Good**:
```python
"""
You are CORTEX - a local, privacy-first AI assistant.

Your role:
- Act as an intelligent office and knowledge assistant
- Answer clearly, concisely, and professionally
- Use documents ONLY when explicitly relevant
"""
```

❌ **Bad**:
```python
"""
You are an AI assistant. Be helpful and answer questions.
"""
```

### 2. Behavioral Guidelines

✅ **Good**:
```python
"""
When answering questions:
1. Check if documents are relevant
2. Quote sources when using document information
3. Admit when you don't know something
4. Keep responses concise unless detail is requested
"""
```

❌ **Bad**:
```python
"""
Try to be helpful.
"""
```

### 3. Identity Boundaries

✅ **Good**:
```python
"""
Do NOT claim to be a human, student, employee, or real-world individual.
If asked about personal experiences, clarify you are an AI.
"""
```

❌ **Bad**:
```python
"""
Pretend to be a helpful person.
"""
```

## Testing Persona Configurations

### 1. Identity Test Suite

```python
# tests/test_persona.py
import unittest
from persona import CORTEX_SYSTEM_PROMPT

class TestPersonaIdentity(unittest.TestCase):
    def test_has_name(self):
        """Verify persona includes name"""
        self.assertIn("CORTEX", CORTEX_SYSTEM_PROMPT)
    
    def test_privacy_emphasis(self):
        """Verify privacy is mentioned"""
        self.assertIn("privacy", CORTEX_SYSTEM_PROMPT.lower())
    
    def test_no_human_claims(self):
        """Verify no claims of being human"""
        self.assertIn("Do NOT claim to be a human", CORTEX_SYSTEM_PROMPT)
    
    def test_role_defined(self):
        """Verify role is clearly defined"""
        self.assertIn("role", CORTEX_SYSTEM_PROMPT.lower())
```

### 2. Behavioral Test

```python
def test_persona_behavior():
    """Test that persona produces expected behavior"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    
    llm = ChatOpenAI(temperature=0)
    
    # Test self-identification
    messages = [
        SystemMessage(content=CORTEX_SYSTEM_PROMPT),
        HumanMessage(content="Who are you?")
    ]
    response = llm.invoke(messages)
    
    assert "CORTEX" in response.content
    assert "AI assistant" in response.content.lower()
    assert "privacy" in response.content.lower() or "local" in response.content.lower()
```

### 3. A/B Testing Framework

```python
# persona_testing.py
from typing import List, Dict
import random

class PersonaABTest:
    def __init__(self, variants: Dict[str, str]):
        """
        Args:
            variants: Dict mapping variant name to prompt
        """
        self.variants = variants
        self.results = {name: [] for name in variants}
    
    def get_variant(self, user_id: str) -> tuple[str, str]:
        """Consistently assign user to variant"""
        hash_val = hash(user_id) % len(self.variants)
        variant_name = list(self.variants.keys())[hash_val]
        return variant_name, self.variants[variant_name]
    
    def record_feedback(self, variant: str, rating: int):
        """Record user feedback (1-5 stars)"""
        self.results[variant].append(rating)
    
    def get_stats(self) -> Dict[str, float]:
        """Get average rating per variant"""
        return {
            name: sum(ratings) / len(ratings) if ratings else 0
            for name, ratings in self.results.items()
        }

# Usage
ab_test = PersonaABTest({
    'professional': CORTEX_SYSTEM_PROMPT,
    'friendly': CORTEX_CASUAL_PROMPT
})

variant_name, prompt = ab_test.get_variant(user_id="user123")
# Use prompt...
ab_test.record_feedback(variant_name, rating=5)
```

## Common Pitfalls and Solutions

### Pitfall 1: Overly Complex Prompts

❌ **Problem**:
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX, an advanced artificial intelligence system designed with 
cutting-edge machine learning algorithms and natural language processing 
capabilities, engineered to provide comprehensive assistance across a wide 
spectrum of domains including but not limited to...
[200 more words]
"""
```

✅ **Solution**:
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local AI assistant.

Focus areas: office work, document analysis, knowledge tasks
Communication: clear, concise, professional
"""
```

### Pitfall 2: Contradictory Instructions

❌ **Problem**:
```python
CORTEX_SYSTEM_PROMPT = """
Be extremely detailed in your responses.
Keep answers concise and brief.
"""
```

✅ **Solution**:
```python
CORTEX_SYSTEM_PROMPT = """
Answer concisely by default.
Provide detailed explanations when specifically requested.
"""
```

### Pitfall 3: Vague Identity

❌ **Problem**:
```python
CORTEX_SYSTEM_PROMPT = """
You are a helpful assistant.
"""
```

✅ **Solution**:
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local, privacy-first AI assistant designed for 
office productivity and knowledge work.
"""
```

### Pitfall 4: Missing Ethical Boundaries

❌ **Problem**:
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX. Answer all questions.
"""
```

✅ **Solution**:
```python
CORTEX_SYSTEM_PROMPT = """
You are CORTEX - a local AI assistant.

Do NOT:
- Claim to be human or a real person
- Provide medical, legal, or financial advice
- Generate harmful or misleading content
"""
```

## Integration Examples

### With Router System

```python
# query.py
from persona import CORTEX_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

def run_chat(query: str, callbacks=None):
    """Chat handler with persona"""
    messages = [
        SystemMessage(content=CORTEX_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    # Execute with LLM...

def run_rag(query: str, callbacks=None):
    """RAG handler with persona and context"""
    docs = retrieve_documents(query)
    
    prompt = f"""{CORTEX_SYSTEM_PROMPT}

Relevant documents:
{format_docs(docs)}

Query: {query}"""
    # Execute with LLM...

def run_meta(query: str, callbacks=None):
    """Meta handler - questions about CORTEX itself"""
    meta_prompt = f"""{CORTEX_SYSTEM_PROMPT}

Additional context:
- You route queries based on intent (RAG for documents, CHAT for conversation)
- You run locally to ensure privacy
- You can search through user documents when relevant

Query: {query}"""
    # Execute with LLM...
```

### With Streaming

```python
# streaming_with_persona.py
from persona import CORTEX_SYSTEM_PROMPT
from streaming import StreamHandler

def stream_with_persona(query: str, on_token):
    """Stream response with CORTEX persona"""
    handler = StreamHandler(on_token=on_token)
    
    messages = [
        SystemMessage(content=CORTEX_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    llm = ChatOpenAI(streaming=True, callbacks=[handler])
    return llm.invoke(messages)
```

## Monitoring and Analytics

### Prompt Effectiveness Metrics

```python
# analytics/persona_metrics.py
from typing import List, Dict
import json

class PersonaMetrics:
    def __init__(self):
        self.interactions = []
    
    def log_interaction(
        self,
        prompt_version: str,
        query: str,
        response: str,
        user_satisfied: bool
    ):
        """Log interaction for analysis"""
        self.interactions.append({
            'prompt_version': prompt_version,
            'query': query,
            'response': response,
            'satisfied': user_satisfied,
            'timestamp': time.time()
        })
    
    def get_satisfaction_rate(self, version: str) -> float:
        """Calculate satisfaction rate for prompt version"""
        version_interactions = [
            i for i in self.interactions 
            if i['prompt_version'] == version
        ]
        
        if not version_interactions:
            return 0.0
        
        satisfied = sum(1 for i in version_interactions if i['satisfied'])
        return satisfied / len(version_interactions)
    
    def export_to_file(self, filepath: str):
        """Export metrics for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.interactions, f, indent=2)
```

## Future Enhancements

### 1. User Preference Learning

```python
# persona_adaptive.py
class AdaptivePersona:
    def __init__(self, base_prompt: str):
        self.base_prompt = base_prompt
        self.user_preferences = {}
    
    def learn_from_feedback(self, user_id: str, feedback: Dict):
        """Adapt persona based on user feedback"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Update preferences
        if feedback.get('too_verbose'):
            self.user_preferences[user_id]['verbosity'] = 'low'
        if feedback.get('needs_more_detail'):
            self.user_preferences[user_id]['verbosity'] = 'high'
    
    def get_personalized_prompt(self, user_id: str) -> str:
        """Get persona prompt tailored to user"""
        prefs = self.user_preferences.get(user_id, {})
        
        prompt = self.base_prompt
        
        if prefs.get('verbosity') == 'low':
            prompt += "\n- Keep responses extremely concise"
        elif prefs.get('verbosity') == 'high':
            prompt += "\n- Provide detailed explanations"
        
        return prompt
```

### 2. Context-Aware Persona

```python
def get_contextual_persona(
    time_of_day: str,
    user_role: str,
    task_type: str
) -> str:
    """Adapt persona based on context"""
    
    base = CORTEX_SYSTEM_PROMPT
    
    # Time-based adjustments
    if time_of_day == "morning":
        base += "\n- Prioritize helping user plan their day"
    
    # Role-based adjustments
    if user_role == "developer":
        base += "\n- Include code examples when relevant"
    elif user_role == "manager":
        base += "\n- Focus on high-level summaries"
    
    # Task-based adjustments
    if task_type == "research":
        base += "\n- Provide citations and sources"
    
    return base
```

## Conclusion

The `persona.py` module is the identity foundation of your AI system. A well-crafted persona:

- **Builds trust** through consistent behavior
- **Sets expectations** for what the system can do
- **Ensures safety** through clear boundaries
- **Improves UX** with appropriate tone and style

### Key Takeaways

1. **Keep it simple**: Clear, concise prompts work better than complex ones
2. **Be specific**: Define exact behaviors and boundaries
3. **Test thoroughly**: Verify persona produces expected behavior
4. **Iterate based on feedback**: Refine based on real usage
5. **Version control**: Track changes and their impact

The current CORTEX persona is minimal and focused, making it easy to understand and maintain while providing a solid foundation for future enhancements.