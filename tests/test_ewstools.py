"""Tests for `ewstools` package."""
import pytest
from ewstools import ewstools as ews


def test_convert(capsys):
    """Correct my_name argument prints"""
    ews.convert("Jill")
    captured = capsys.readouterr()
    assert "Jill" in captured.out
    
    
def test_

    
    
    