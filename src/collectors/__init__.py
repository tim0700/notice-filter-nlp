"""Collector package exposing BaseCollector and collector registry."""

from src.collectors.base import COLLECTOR_REGISTRY, BaseCollector

__all__ = ["BaseCollector", "COLLECTOR_REGISTRY"]
