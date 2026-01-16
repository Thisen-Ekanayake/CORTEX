import os
import joblib
import numpy as np
from enum import Enum
from typing import Tuple, Dict

from cortex.query import run_rag, run_meta, run_chat


class Route(Enum):
    RAG = "rag"
    CHAT = "chat"
    META = "meta"


class TFIDFRouter:
    """
    TF-IDF based query router that classifies queries into RAG, CHAT, or META categories.
    """
    
    # Map numeric labels to Route enums
    LABEL_TO_ROUTE = {
        0: Route.CHAT,
        1: Route.META,
        2: Route.RAG,
    }
    
    def __init__(self, model_dir: str = "tf-idf_classifier/model"):
        """
        Initialize the TF-IDF router by loading trained models.
        
        Args:
            model_dir: Directory containing the trained tfidf.joblib and classifier.joblib
        
        Raises:
            FileNotFoundError: If model files are not found
        """
        vectorizer_path = os.path.join(model_dir, "tfidf.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found at: {vectorizer_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found at: {classifier_path}")
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
    
    def route_query(self, query: str) -> Tuple[Route, Dict[str, float]]:
        """
        Classify query and return route with confidence scores.
        
        Args:
            query: User's question/message
        
        Returns:
            tuple: (predicted_route, confidence_scores)
                - predicted_route: Route enum for the predicted category
                - confidence_scores: Dict mapping route names to confidence scores (0-1)
        
        Example:
            >>> router = TFIDFRouter()
            >>> route, scores = router.route_query("What can you do?")
            >>> print(route)  # Route.META
            >>> print(scores)  # {'chat': 0.05, 'meta': 0.92, 'rag': 0.03}
        """
        # Transform query to TF-IDF features
        X = self.vectorizer.transform([query])
        
        # Get predicted class
        predicted_label = self.classifier.predict(X)[0]
        predicted_route = self.LABEL_TO_ROUTE[predicted_label]
        
        # Get confidence scores (probabilities) for all classes
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Map probabilities to route names
        confidence_scores = {
            Route.CHAT.value: float(probabilities[0]),
            Route.META.value: float(probabilities[1]),
            Route.RAG.value: float(probabilities[2]),
        }
        
        return predicted_route, confidence_scores


# Global router instance (lazy-loaded)
_router = None


def get_router(model_dir: str = "tf-idf_classifier/model") -> TFIDFRouter:
    """
    Get or create the global TF-IDF router instance.
    
    Args:
        model_dir: Directory containing trained models
    
    Returns:
        TFIDFRouter instance
    """
    global _router
    if _router is None:
        _router = TFIDFRouter(model_dir)
    return _router


def route_query(query: str) -> Tuple[Route, Dict[str, float]]:
    """
    Classify query and return appropriate route with confidence scores.
    
    Args:
        query: User's question/message
    
    Returns:
        tuple: (predicted_route, confidence_scores)
    
    Example:
        >>> route, scores = route_query("How does RAG work?")
        >>> print(f"Route: {route.value}")
        >>> print(f"Confidence: {scores[route.value]:.2%}")
    """
    router = get_router()
    return router.route_query(query)


def execute(query: str, callbacks=None, confidence_threshold: float = 0.5) -> Tuple[str, Route, Dict[str, float]]:
    """
    Execute query based on routing decision.
    
    Args:
        query: User's question/message
        callbacks: Optional callbacks for streaming
        confidence_threshold: Minimum confidence to use predicted route (0-1).
                            If confidence is below threshold, falls back to CHAT.
    
    Returns:
        tuple: (result_string, route, confidence_scores)
            - result_string: The response from the appropriate handler
            - route: Route enum that was actually used
            - confidence_scores: Dict of confidence scores for all categories
    
    Example:
        >>> result, route, scores = execute("What's in the Q3 report?")
        >>> print(f"Used route: {route.value}")
        >>> print(f"Confidence scores: {scores}")
        >>> print(f"Result: {result}")
    """
    predicted_route, confidence_scores = route_query(query)
    
    # Get confidence for the predicted route
    predicted_confidence = confidence_scores[predicted_route.value]
    
    # Fall back to CHAT if confidence is too low
    if predicted_confidence < confidence_threshold:
        route = Route.CHAT
        result = run_chat(query, callbacks=callbacks)
        return result, route, confidence_scores
    
    # Execute based on predicted route
    route = predicted_route
    
    if route == Route.RAG:
        result = run_rag(query, callbacks=callbacks)
        
        # If no documents found, fall back to CHAT
        if result is None:
            route = Route.CHAT
            result = run_chat(query, callbacks=callbacks)
        
        return result, route, confidence_scores
    
    elif route == Route.META:
        result = run_meta(query, callbacks=callbacks)
        return result, route, confidence_scores
    
    else:  # Route.CHAT
        result = run_chat(query, callbacks=callbacks)
        return result, route, confidence_scores


# Convenience function for debugging/testing
def print_routing_info(query: str) -> None:
    """
    Print detailed routing information for a query (useful for debugging).
    
    Args:
        query: User's question/message
    
    Example:
        >>> print_routing_info("What does the report say about sales?")
        Query: "What does the report say about sales?"
        Predicted Route: rag
        Confidence Scores:
          - chat: 5.23%
          - meta: 2.15%
          - rag: 92.62%
    """
    route, scores = route_query(query)
    
    print(f'Query: "{query}"')
    print(f"Predicted Route: {route.value}")
    print("Confidence Scores:")
    
    # Sort by confidence (highest first)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for category, confidence in sorted_scores:
        print(f"  - {category}: {confidence:.2%}")