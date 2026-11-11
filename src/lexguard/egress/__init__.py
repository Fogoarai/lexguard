"""The trust-boundary gateway: nothing reaches a language model except through it."""

from lexguard.egress.gateway import EgressGateway, Scanner

__all__ = ["EgressGateway", "Scanner"]
