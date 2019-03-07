"""Tests for `notebookc` package."""
import pytest
from ewstools import ewstools


def test_convert(capsys):
    """Correct my_name argument prints"""
    ewstools.convert("Jill")
    captured = capsys.readouterr()
    assert "Jall" in captured.out