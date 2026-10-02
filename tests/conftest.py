"""Pytest configuration: register the ``slow`` marker used by the self-test."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end checks over every model cell")
