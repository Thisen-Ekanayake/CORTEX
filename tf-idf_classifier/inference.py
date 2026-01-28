import os
import joblib

# Option A labels
LABEL_MAP = {
    "chat": 0,
    "meta": 1,
    "rag": 2,        # optional generic rag (keep if you trained with it)
    "rag_doc": 3,    # document search
    "rag_img": 4,    # image search
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class TextClassifier:
    def __init__(self, model_dir="tf-idf_classifier/model"):
        """Load the trained vectorizer and classifier."""
        vectorizer_path = os.path.join(model_dir, "tfidf.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)

        # IMPORTANT: actual class id order used by the trained classifier
        # e.g. array([0,1,2,3,4]) but don't assume.
        self.class_ids = list(self.classifier.classes_)

        # Validate we can map all model classes back to labels
        unknown = [cid for cid in self.class_ids if cid not in ID_TO_LABEL]
        if unknown:
            raise ValueError(
                f"Model contains unknown class ids: {unknown}. "
                f"Update LABEL_MAP/ID_TO_LABEL to match training."
            )

        print(f"Model loaded from {model_dir}")
        print(f"Model classes: {[ID_TO_LABEL[c] for c in self.class_ids]}")

    def _probs_to_scores(self, probs):
        """Map probability array to {label: prob} using classifier.classes_ order."""
        return {ID_TO_LABEL[cid]: float(p) for cid, p in zip(self.class_ids, probs)}

    def predict(self, text):
        """
        Predict the category and confidence scores for a given text.
        Returns:
            dict with predicted_label, confidence_scores, predicted_class_id
        """
        X = self.vectorizer.transform([text])

        probs = self.classifier.predict_proba(X)[0]
        pred_class_id = int(self.classifier.predict(X)[0])

        confidence_scores = self._probs_to_scores(probs)
        predicted_label = ID_TO_LABEL[pred_class_id]

        return {
            "predicted_label": predicted_label,
            "confidence_scores": confidence_scores,
            "predicted_class_id": pred_class_id,
        }

    def predict_batch(self, texts):
        """
        Predict categories for multiple texts.
        Returns:
            list of dicts
        """
        X = self.vectorizer.transform(texts)
        probs_all = self.classifier.predict_proba(X)
        preds = self.classifier.predict(X)

        results = []
        for text, pred_id, probs in zip(texts, preds, probs_all):
            confidence_scores = self._probs_to_scores(probs)
            pred_id = int(pred_id)
            results.append({
                "text": text,
                "predicted_label": ID_TO_LABEL[pred_id],
                "confidence_scores": confidence_scores,
                "predicted_class_id": pred_id,
            })

        return results

    def rag_total_score(self, confidence_scores):
        """
        Optional helper: combine rag-like classes into one score.
        Useful if downstream just needs 'rag vs not rag' sometimes.
        """
        return (
            confidence_scores.get("rag", 0.0) +
            confidence_scores.get("rag_doc", 0.0) +
            confidence_scores.get("rag_img", 0.0)
        )


def print_results(result):
    """Pretty print the classification results with explicit RAG breakdown."""
    scores = result["confidence_scores"]

    print("\n" + "=" * 60)
    print(f"Predicted Category: {result['predicted_label'].upper()}")
    print("=" * 60)

    print("\nRAG Confidence Breakdown:")
    print("-" * 60)
    print(f"{'rag':>8}:     {scores.get('rag', 0.0):.2%}")
    print(f"{'rag_doc':>8}: {scores.get('rag_doc', 0.0):.2%}")
    print(f"{'rag_img':>8}: {scores.get('rag_img', 0.0):.2%}")

    print("\nOther Classes:")
    print("-" * 60)
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if label in {"rag", "rag_doc", "rag_img"}:
            continue
        print(f"{label:>8}: {score:.2%}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    classifier = TextClassifier(model_dir="tf-idf_classifier/model")

    test_prompts = [
        # chat
        "Tell me a joke about programmers and coffee.",

        # meta
        "What tools do you have access to in this assistant pipeline?",

        # rag_doc
        "What are the documented differences between encoder-only and decoder-only transformers in published research?",
        "Find papers comparing LoRA and full fine-tuning with reported accuracy and compute costs.",

        # rag_img
        "Show example MRI images of meningioma versus glioma with visual differences highlighted.",
        "Find diagrams illustrating attention mechanisms in transformers for a presentation slide.",

        # generic rag (optional)
        "Search the knowledge base for the refund policy and quote the exact clause.",
    ]

    print("\nSingle Prediction Examples:")
    for prompt in test_prompts:
        print(f"\nInput: '{prompt}'")
        result = classifier.predict(prompt)
        print_results(result)
