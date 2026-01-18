"""
Reinforcement Learning Router for CORTEX
Learns from user corrections to improve routing accuracy over time.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from collections import deque

from cortex.router import Route, TFIDFRouter


@dataclass
class RoutingFeedback:
    """Single routing feedback entry"""
    query: str
    predicted_route: str
    actual_route: str
    confidence_scores: Dict[str, float]
    correct: bool
    timestamp: str
    reward: float


class RLRouter:
    """
    Reinforcement Learning Router that learns from user feedback.
    
    Uses the base TF-IDF classifier but adjusts predictions based on
    accumulated feedback and applies online learning.
    """
    
    def __init__(self, 
                 base_router: TFIDFRouter,
                 feedback_dir: str = "rl_feedback",
                 learning_rate: float = 0.1,
                 memory_size: int = 1000):
        """
        Initialize RL Router.
        
        Args:
            base_router: The underlying TF-IDF router
            feedback_dir: Directory to store feedback and metrics
            learning_rate: How quickly to adjust from feedback (0-1)
            memory_size: Max number of recent feedbacks to keep in memory
        """
        self.base_router = base_router
        self.feedback_dir = feedback_dir
        self.learning_rate = learning_rate
        
        # Create feedback directory if it doesn't exist
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Feedback storage
        self.feedback_history = deque(maxlen=memory_size)
        self.feedback_file = os.path.join(feedback_dir, "feedback.jsonl")
        self.metrics_file = os.path.join(feedback_dir, "metrics.json")
        
        # Performance metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.route_accuracy = {
            Route.RAG.value: {"correct": 0, "total": 0},
            Route.META.value: {"correct": 0, "total": 0},
            Route.CHAT.value: {"correct": 0, "total": 0},
        }
        
        # Confidence adjustment weights (learned from feedback)
        self.confidence_weights = {
            Route.RAG.value: 1.0,
            Route.META.value: 1.0,
            Route.CHAT.value: 1.0,
        }
        
        # Load existing feedback and metrics
        self._load_feedback()
        self._load_metrics()
    
    def route_query(self, query: str) -> Tuple[Route, Dict[str, float]]:
        """
        Route query using base router with RL adjustments.
        
        Args:
            query: User's question
        
        Returns:
            tuple: (predicted_route, adjusted_confidence_scores)
        """
        # Get base prediction
        base_route, base_scores = self.base_router.route_query(query)
        
        # Apply learned confidence adjustments
        adjusted_scores = {}
        for route_name, base_confidence in base_scores.items():
            weight = self.confidence_weights.get(route_name, 1.0)
            adjusted_scores[route_name] = base_confidence * weight
        
        # Normalize adjusted scores
        total = sum(adjusted_scores.values())
        if total > 0:
            adjusted_scores = {k: v/total for k, v in adjusted_scores.items()}
        
        # Find route with highest adjusted score
        best_route_name = max(adjusted_scores.items(), key=lambda x: x[1])[0]
        predicted_route = Route(best_route_name)
        
        return predicted_route, adjusted_scores
    
    def record_feedback(self, 
                       query: str,
                       predicted_route: Route,
                       actual_route: Route,
                       confidence_scores: Dict[str, float]) -> RoutingFeedback:
        """
        Record user feedback and update the model.
        
        Args:
            query: The user's query
            predicted_route: What the model predicted
            actual_route: What the user selected
            confidence_scores: Model's confidence for each route
        
        Returns:
            RoutingFeedback object with reward information
        """
        correct = (predicted_route == actual_route)
        
        # Calculate reward
        if correct:
            # Reward based on confidence (higher confidence = better)
            base_reward = 1.0
            confidence_bonus = confidence_scores[actual_route.value] * 0.5
            reward = base_reward + confidence_bonus
        else:
            # Penalty based on how wrong we were
            base_penalty = -1.0
            confidence_penalty = confidence_scores[predicted_route.value] * 0.5
            reward = base_penalty - confidence_penalty
        
        # Create feedback entry
        feedback = RoutingFeedback(
            query=query,
            predicted_route=predicted_route.value,
            actual_route=actual_route.value,
            confidence_scores=confidence_scores,
            correct=correct,
            timestamp=datetime.now().isoformat(),
            reward=reward
        )
        
        # Update metrics
        self._update_metrics(feedback)
        
        # Apply learning
        self._learn_from_feedback(feedback)
        
        # Store feedback
        self.feedback_history.append(feedback)
        self._save_feedback(feedback)
        
        return feedback
    
    def _learn_from_feedback(self, feedback: RoutingFeedback):
        """
        Update confidence weights based on feedback.
        
        This implements a simple Q-learning style update where we adjust
        the confidence weights for each route based on success/failure.
        """
        if feedback.correct:
            # Increase weight for correct route (but cap at 2.0)
            current = self.confidence_weights[feedback.actual_route]
            self.confidence_weights[feedback.actual_route] = min(
                2.0,
                current + self.learning_rate * feedback.reward
            )
        else:
            # Decrease weight for incorrectly predicted route
            current = self.confidence_weights[feedback.predicted_route]
            self.confidence_weights[feedback.predicted_route] = max(
                0.5,  # Don't go below 0.5
                current + self.learning_rate * feedback.reward
            )
            
            # Slightly increase weight for correct route
            current_correct = self.confidence_weights[feedback.actual_route]
            self.confidence_weights[feedback.actual_route] = min(
                2.0,
                current_correct + self.learning_rate * abs(feedback.reward) * 0.5
            )
        
        # Save updated weights
        self._save_metrics()
    
    def _update_metrics(self, feedback: RoutingFeedback):
        """Update performance metrics."""
        self.total_predictions += 1
        
        if feedback.correct:
            self.correct_predictions += 1
        
        # Update per-route accuracy
        route_stats = self.route_accuracy[feedback.actual_route]
        route_stats["total"] += 1
        if feedback.correct:
            route_stats["correct"] += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current performance metrics.
        
        Returns:
            Dict with overall and per-route accuracy
        """
        overall_accuracy = (
            self.correct_predictions / self.total_predictions 
            if self.total_predictions > 0 else 0.0
        )
        
        route_metrics = {}
        for route, stats in self.route_accuracy.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                route_metrics[route] = {
                    "accuracy": accuracy,
                    "total": stats["total"],
                    "correct": stats["correct"]
                }
            else:
                route_metrics[route] = {
                    "accuracy": 0.0,
                    "total": 0,
                    "correct": 0
                }
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "route_accuracy": route_metrics,
            "confidence_weights": self.confidence_weights,
            "recent_feedbacks": len(self.feedback_history)
        }
    
    def _save_feedback(self, feedback: RoutingFeedback):
        """Append feedback to JSONL file."""
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(asdict(feedback)) + '\n')
    
    def _load_feedback(self):
        """Load recent feedback from file."""
        if not os.path.exists(self.feedback_file):
            return
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    feedback = RoutingFeedback(**data)
                    self.feedback_history.append(feedback)
                except (json.JSONDecodeError, TypeError):
                    continue
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics = self.get_metrics()
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _load_metrics(self):
        """Load metrics from file."""
        if not os.path.exists(self.metrics_file):
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            self.total_predictions = data.get("total_predictions", 0)
            self.correct_predictions = data.get("correct_predictions", 0)
            
            # Load route accuracy
            route_acc = data.get("route_accuracy", {})
            for route, stats in route_acc.items():
                if route in self.route_accuracy:
                    self.route_accuracy[route] = {
                        "correct": stats.get("correct", 0),
                        "total": stats.get("total", 0)
                    }
            
            # Load confidence weights
            weights = data.get("confidence_weights", {})
            for route, weight in weights.items():
                if route in self.confidence_weights:
                    self.confidence_weights[route] = weight
                    
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    def get_recent_feedback(self, n: int = 10) -> list:
        """
        Get the n most recent feedback entries.
        
        Args:
            n: Number of recent entries to return
        
        Returns:
            List of RoutingFeedback objects
        """
        return list(self.feedback_history)[-n:]
    
    def reset_learning(self):
        """Reset all learning (weights back to 1.0, clear metrics)."""
        self.confidence_weights = {
            Route.RAG.value: 1.0,
            Route.META.value: 1.0,
            Route.CHAT.value: 1.0,
        }
        self.total_predictions = 0
        self.correct_predictions = 0
        self.route_accuracy = {
            Route.RAG.value: {"correct": 0, "total": 0},
            Route.META.value: {"correct": 0, "total": 0},
            Route.CHAT.value: {"correct": 0, "total": 0},
        }
        self._save_metrics()


# Global RL router instance
_rl_router = None


def get_rl_router(feedback_dir: str = "rl_feedback",
                  learning_rate: float = 0.1) -> RLRouter:
    """
    Get or create the global RL router instance.
    
    Args:
        feedback_dir: Directory to store feedback
        learning_rate: Learning rate for RL updates
    
    Returns:
        RLRouter instance
    """
    global _rl_router
    if _rl_router is None:
        from cortex.router import get_router
        base_router = get_router()
        _rl_router = RLRouter(
            base_router=base_router,
            feedback_dir=feedback_dir,
            learning_rate=learning_rate
        )
    return _rl_router