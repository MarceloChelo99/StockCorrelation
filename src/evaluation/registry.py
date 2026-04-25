"""Explicit evaluator registry."""
from __future__ import annotations

from src.evaluation.clustering import ClusteringEvaluator
from src.evaluation.covariance import CovarianceEvaluator
from src.evaluation.multiview_covariance import MultiviewCovarianceEvaluator
from src.evaluation.multiview_peers import MultiViewPeerEvaluator
from src.evaluation.peers import PeersEvaluator
from src.evaluation.relationships import RelationshipGraphEvaluator
from src.evaluation.view_comparison import ViewComparisonEvaluator


EVALUATOR_REGISTRY = {
    "clustering": ClusteringEvaluator(),
    "peers": PeersEvaluator(),
    "covariance": CovarianceEvaluator(),
    "relationships": RelationshipGraphEvaluator(),
    "multiview_covariance": MultiviewCovarianceEvaluator(),
    "multiview_peers": MultiViewPeerEvaluator(),
    "view_comparison": ViewComparisonEvaluator(),
}
