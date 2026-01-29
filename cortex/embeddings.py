from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None

def get_embeddings():
    """
    Get or initialize the global embeddings model.
    
    Uses a singleton pattern to ensure the embeddings model is only loaded once.
    The model used is 'all-MiniLM-L6-v2' from HuggingFace, which provides
    384-dimensional sentence embeddings.
    
    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model instance.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    return _embeddings