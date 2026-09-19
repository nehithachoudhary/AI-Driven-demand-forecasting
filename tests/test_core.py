import json
import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.inventory import inventory_recommendation
from src.preprocessing import load_and_validate

def test_dataset_loads_without_mutation():
    df, report = load_and_validate()
    assert report["rows"] == 76000
    assert len(df) == 76000

def test_inventory_formulas():
    result = inventory_recommendation(100, 80, 10, lead_time=7, service_level=.95)
    assert result["safety_stock"] > 0
    assert result["reorder_point"] > result["safety_stock"]
    assert result["suggested_reorder_quantity"] >= 0

def test_api_validation_without_artifacts(monkeypatch):
    import app
    response = app.app.test_client().post('/api/forecast', json={"store_id":"UNKNOWN","product_id":"P0001","horizon":7})
    assert response.status_code == 400

def test_invalid_horizon():
    import app
    response = app.app.test_client().post('/api/forecast', json={"store_id":"S001","product_id":"P0001","horizon":8})
    assert response.status_code == 400