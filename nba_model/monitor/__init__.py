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
    >>> from nba_model.monitor import DriftDetector, RetrainingTrigger
    >>> detector = DriftDetector(reference_data)
    >>> result = detector.check_drift(recent_data)
    >>> if result.has_drift:
    ...     print("Retraining recommended")
"""

from __future__ import annotations

from nba_model.monitor.drift import (
    MONITORED_FEATURES,
    ConceptDriftDetector,
    ConceptDriftResult,
    DriftCheckResult,
    DriftDetectionError,
    DriftDetector,
    FeatureDriftDetail,
    InsufficientDataError,
    KSTestResult,
)
from nba_model.monitor.triggers import (
    RetrainingTrigger,
    TriggerContext,
    TriggerResult,
)
from nba_model.monitor.versioning import (
    ModelVersionManager,
    VersionComparisonResult,
    VersionMetadata,
)

__all__ = [
    "MONITORED_FEATURES",
    "ConceptDriftDetector",
    "ConceptDriftResult",
    "DriftCheckResult",
    "DriftDetectionError",
    # drift.py
    "DriftDetector",
    "FeatureDriftDetail",
    "InsufficientDataError",
    "KSTestResult",
    # versioning.py
    "ModelVersionManager",
    # triggers.py
    "RetrainingTrigger",
    "TriggerContext",
    "TriggerResult",
    "VersionComparisonResult",
    "VersionMetadata",
]
