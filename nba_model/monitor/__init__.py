"""Model monitoring and self-improvement for NBA model.

This module provides drift detection, retraining triggers, and model
versioning for continuous model quality maintenance.

Submodules:
    drift: Covariate and concept drift detection
    triggers: Retraining trigger logic
    versioning: Model version management

Key concepts:
    - Covariate drift: Input feature distributions shift
    - Concept drift: Feature-target relationship changes
    - PSI (Population Stability Index): Drift severity metric
    - Retraining triggers: Automated model refresh decisions

Example:
    >>> from nba_model.monitor import DriftDetector
    >>> detector = DriftDetector(reference_data)
    >>> result = detector.check_drift(recent_data)
    >>> if result['has_drift']:
    ...     print("Retraining recommended")
"""

from __future__ import annotations

# Public API - will be populated in Phase 6
__all__: list[str] = []
