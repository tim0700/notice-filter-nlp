"""Collector package exposing BaseCollector, concrete collectors, registry, and runner."""

from src.collectors.base import COLLECTOR_REGISTRY, BaseCollector
from src.collectors.goodjob import (
    GoodjobCollector,
    GoodjobExtracurricularCollector,
    GoodjobGeneralCollector,
    GoodjobRecommendCollector,
)
from src.collectors.runner import collect_all
from src.collectors.seongnam import SeongnamYouthCollector
from src.collectors.swcon import (
    SwconAcademicCollector,
    SwconCareerCollector,
    SwconCollector,
)
from src.collectors.youth_api import YouthApiCollector

__all__ = [
    "BaseCollector",
    "COLLECTOR_REGISTRY",
    "SwconCollector",
    "SwconAcademicCollector",
    "SwconCareerCollector",
    "GoodjobCollector",
    "GoodjobExtracurricularCollector",
    "GoodjobGeneralCollector",
    "GoodjobRecommendCollector",
    "SeongnamYouthCollector",
    "YouthApiCollector",
    "collect_all",
]
