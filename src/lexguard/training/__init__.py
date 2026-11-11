"""Training-set export with manifests, a DP-capable classifier and memorization tests."""

from lexguard.training.classifier import DpReport, DpSgd, KindClassifier, featurize
from lexguard.training.dataset import DatasetTainted, Example, TrainingSet, TrainingSets
from lexguard.training.manifest import (
    Entry,
    Exclusion,
    Manifest,
    ManifestRegistry,
    Reason,
    Taint,
)
from lexguard.training.memorization import Canary, Exposure, NgramLanguageModel, exposure, plant

__all__ = [
    "Canary",
    "DatasetTainted",
    "DpReport",
    "DpSgd",
    "Entry",
    "Example",
    "Exclusion",
    "Exposure",
    "KindClassifier",
    "Manifest",
    "ManifestRegistry",
    "NgramLanguageModel",
    "Reason",
    "Taint",
    "TrainingSet",
    "TrainingSets",
    "exposure",
    "featurize",
    "plant",
]
