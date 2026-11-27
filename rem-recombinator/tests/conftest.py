"""Shared fixtures. Every database lives under tmp_path."""

from __future__ import annotations

import pytest

from rem_recombinator.config import load_config
from rem_recombinator.store import Store


@pytest.fixture
def config():
    return load_config({})


@pytest.fixture
def store(tmp_path):
    database = Store(tmp_path / "fragments.db")
    yield database
    database.close()
