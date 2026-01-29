import os
import joblib
from enum import Enum
from typing import Tuple, Dict

from cortex.query import run_rag, run_meta, run_chat, run_rag_mode


class Route(Enum):
    """
    Enumeration of possible query routing destinations.
    
    Routes:
        CHAT: General conversational queries
        META: Queries about the system itself
        RAG: Generic retrieval-augmented generation
        RAG_DOC: Document-based retrieval
        RAG_IMG: Image-based retrieval
    """
    CHAT = "chat"
    META = "meta"
    RAG = "rag"          # generic rag entry
    RAG_DOC = "rag_doc"  # document search
    RAG_IMG = "rag_img"  # image search


class TFIDFRouter:
    """
    TF-IDF based query router supporting:
    chat / meta / rag / rag_doc / rag_img
    """

    def __init__(self, model_dir: str = "tf-idf_classifier/model"):
        """
        Initialize the TF-IDF router with pre-trained models.
        
        Loads the TF-IDF vectorizer and classifier from disk. The models
        must be trained and saved using the train_classifier.py script.
        
        Args:
            model_dir: Directory containing tfidf.joblib and classifier.joblib files.
        
        Raises:
            FileNotFoundError: If model files are missing.
            ValueError: If classifier contains unknown class IDs.
        """
        vectorizer_path = os.path.join(model_dir, "tfidf.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found at: {vectorizer_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found at: {classifier_path}")

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)

        # Actual class ids used by the trained model
        self.class_ids = list(self.classifier.classes_)

        # Mapping from class id → Route
        self.id_to_route = {
            0: Route.CHAT,
            1: Route.META,
            2: Route.RAG,
            3: Route.RAG_DOC,
            4: Route.RAG_IMG,
        }

        unknown = [cid for cid in self.class_ids if cid not in self.id_to_route]
        if unknown:
            raise ValueError(f"Unknown class IDs in model: {unknown}")

    def _scores_from_probs(self, probs) -> Dict[str, float]:
        """
        Map probability vector to route-name → score dictionary.
        
        Args:
            probs: Probability array from classifier.predict_proba().
        
        Returns:
            dict: Mapping of route names (str) to confidence scores (float).
        """
        scores = {}
        for cid, prob in zip(self.class_ids, probs):
            route = self.id_to_route[cid]
            scores[route.value] = float(prob)
        return scores

    def route_query(self, query: str) -> Tuple[Route, Dict[str, float]]:
        """
        Classify query and return predicted route with confidence scores.
        
        Args:
            query: User query string to classify.
        
        Returns:
            tuple: (predicted_route, confidence_scores)
                - predicted_route: Route enum value (most likely category)
                - confidence_scores: Dict mapping route names to probabilities
        """
        X = self.vectorizer.transform([query])

        probs = self.classifier.predict_proba(X)[0]
        pred_class_id = int(self.classifier.predict(X)[0])
        pred_route = self.id_to_route[pred_class_id]

        confidence_scores = self._scores_from_probs(probs)

        return pred_route, confidence_scores


# ---- Global router (lazy) ----
_router = None


def get_router(model_dir: str = "tf-idf_classifier/model") -> TFIDFRouter:
    """
    Get or create the global router instance (singleton pattern).
    
    Args:
        model_dir: Directory containing model files (default: "tf-idf_classifier/model").
    
    Returns:
        TFIDFRouter: The global router instance.
    """
    global _router
    if _router is None:
        _router = TFIDFRouter(model_dir)
    return _router


def route_query(query: str) -> Tuple[Route, Dict[str, float]]:
    """
    Route a query using the global router instance.
    
    Convenience function that uses the singleton router to classify a query.
    
    Args:
        query: User query string to classify.
    
    Returns:
        tuple: (predicted_route, confidence_scores)
    """
    router = get_router()
    return router.route_query(query)


def execute(
    query: str,
    callbacks=None,
    confidence_threshold: float = 0.5
) -> Tuple[str, Route, Dict[str, float]]:
    """
    Execute query based on routing decision.
    
    Routes the query to the appropriate handler (RAG, META, or CHAT) based on
    classification confidence. Falls back to CHAT if confidence is below threshold
    or if RAG retrieval fails.
    
    Args:
        query: User query string to process.
        callbacks: Optional callback handlers for streaming responses.
        confidence_threshold: Minimum confidence required to use specialized route (default: 0.5).
    
    Returns:
        tuple: (response_text, actual_route, confidence_scores)
            - response_text: Generated response string
            - actual_route: Route enum that was actually used
            - confidence_scores: Dict of confidence scores for all routes
    """

    predicted_route, scores = route_query(query)

    # ---- RAG total confidence (explicit, visible) ----
    rag_total = (
        scores.get("rag", 0.0) +
        scores.get("rag_doc", 0.0) +
        scores.get("rag_img", 0.0)
    )

    # ---- Decide high-level route ----
    if predicted_route in {Route.RAG, Route.RAG_DOC, Route.RAG_IMG}:
        if rag_total < confidence_threshold:
            result = run_chat(query, callbacks=callbacks)
            return result, Route.CHAT, scores

        # ---- Sub-routing inside RAG ----
        if predicted_route == Route.RAG_IMG:
            result = run_rag_mode(query, callbacks=callbacks, mode="image")
        else:
            # rag or rag_doc
            result = run_rag_mode(query, callbacks=callbacks, mode="document")

        if result is None:
            result = run_chat(query, callbacks=callbacks)
            return result, Route.CHAT, scores

        return result, predicted_route, scores

    elif predicted_route == Route.META:
        if scores["meta"] < confidence_threshold:
            result = run_chat(query, callbacks=callbacks)
            return result, Route.CHAT, scores

        result = run_meta(query, callbacks=callbacks)
        return result, Route.META, scores

    else:  # CHAT
        result = run_chat(query, callbacks=callbacks)
        return result, Route.CHAT, scores


# ---- Debug helper ----
def print_routing_info(query: str) -> None:
    """
    Print detailed routing information for a query (debug helper).
    
    Displays the query, predicted route, individual confidence scores,
    and total RAG confidence (sum of rag, rag_doc, rag_img).
    
    Args:
        query: User query string to analyze.
    """
    route, scores = route_query(query)

    print(f'Query: "{query}"')
    print(f"Predicted Route: {route.value}")
    print("Confidence Scores:")

    for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k}: {v:.2%}")

    rag_total = (
        scores.get("rag", 0.0) +
        scores.get("rag_doc", 0.0) +
        scores.get("rag_img", 0.0)
    )
    print(f"  → RAG total: {rag_total:.2%}")
