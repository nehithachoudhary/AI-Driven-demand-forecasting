import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from app import app


@pytest.fixture()
def client(tmp_path):
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    app.instance_path = str(tmp_path)
    app.extensions.pop("db", None)
    with app.test_client() as test_client:
        with app.app_context():
            from src.database import init_db
            init_db()
        yield test_client


def registration(email, name):
    return {"full_name": name, "business_name": f"{name} Traders", "email": email, "phone": "9876543210", "address": "Market Road", "city": "Hyderabad", "state": "Telangana", "region": "South", "security_question": "What is your favorite product?", "security_answer": "Jaggery", "password": "SecurePass123", "confirm_password": "SecurePass123"}


def login(client, email):
    return client.post("/login", data={"email": email, "password": "SecurePass123"}, follow_redirects=True)


def test_registration_login_and_protected_route(client):
    assert client.get("/dashboard").status_code == 302
    assert client.post("/register", data=registration("a@example.com", "Asha")).status_code == 302
    response = login(client, "a@example.com")
    assert response.status_code == 200
    assert b"Welcome back" in response.data


def test_duplicate_email_and_wrong_password(client):
    client.post("/register", data=registration("same@example.com", "Asha"))
    response = client.post("/register", data=registration("same@example.com", "Bina"))
    assert response.status_code == 200
    response = client.post("/login", data={"email": "same@example.com", "password": "wrongpass"})
    assert b"Invalid email" in response.data


def test_user_data_isolation(client):
    client.post("/register", data=registration("a@example.com", "Asha")); login(client, "a@example.com")
    created = client.post("/api/products", json={"name": "Onion", "category": "Produce", "unit": "kg", "region": "South", "current_stock": 100, "safety_stock": 20, "lead_time_days": 3})
    product_id = created.get_json()["id"]
    client.get("/logout")
    client.post("/register", data=registration("b@example.com", "Bina")); login(client, "b@example.com")
    assert client.get("/api/products").get_json() == []
    assert client.put(f"/api/products/{product_id}", json={"name": "Stolen", "unit": "kg", "region": "South"}).status_code == 404


def test_sale_updates_inventory_and_rejects_oversell(client):
    client.post("/register", data=registration("a@example.com", "Asha")); login(client, "a@example.com")
    product_id = client.post("/api/products", json={"name": "Rice", "unit": "kg", "region": "South", "current_stock": 10, "safety_stock": 2, "lead_time_days": 3}).get_json()["id"]
    response = client.post("/sales/new", data={"product_id": product_id, "quantity": 20}, follow_redirects=True)
    assert b"exceeds available stock" in response.data
    response = client.post("/sales/new", data={"product_id": product_id, "quantity": 4, "sale_date": "2026-09-17"}, follow_redirects=True)
    assert b"Sale recorded" in response.data
    assert client.get("/api/inventory").get_json()[0]["current_stock"] == 6


def test_customer_forecast_is_honest_about_insufficient_history(client):
    client.post("/register", data=registration("a@example.com", "Asha")); login(client, "a@example.com")
    product_id = client.post("/api/products", json={"name": "Turmeric", "unit": "kg", "region": "South"}).get_json()["id"]
    response = client.get(f"/api/forecasts/{product_id}")
    assert response.status_code == 200
    assert response.get_json()["status"] == "not_trained"
    assert response.get_json()["required_records"] == 54


def test_customer_training_endpoint_rejects_insufficient_history(client):
    client.post("/register", data=registration("customer@example.com", "Cathy")); login(client, "customer@example.com")
    product_id = client.post("/api/products", json={"name": "Jaggery", "unit": "kg", "region": "South"}).get_json()["id"]
    response = client.post(f"/api/forecasts/{product_id}/train")
    assert response.status_code == 422
    assert "Current records: 0" in response.get_json()["error"]


def test_customer_forecast_errors_are_json_only(client):
    client.post("/register", data=registration("json@example.com", "Json")); login(client, "json@example.com")
    product_id = client.post("/api/products", json={"name": "Cardamom", "unit": "kg", "region": "South"}).get_json()["id"]
    response = client.post(f"/api/forecasts/{product_id}/generate", json={"horizon": 9})
    assert response.status_code == 422
    assert response.is_json is True
    assert "Forecast horizon" in response.get_json()["error"]


def test_password_mismatch_preserves_non_sensitive_fields(client):
    form = registration("mismatch@example.com", "Maya")
    form["confirm_password"] = "different"
    response = client.post("/register", data=form)
    assert b"Passwords do not match" in response.data
    assert b"Maya" in response.data
    assert b"Maya Traders" in response.data
    assert b"mismatch@example.com" in response.data
    assert b"SecurePass123" not in response.data


def test_suggestion_is_actual_history_only_and_does_not_create_sale(client):
    client.post("/register", data=registration("suggest@example.com", "Sana")); login(client, "suggest@example.com")
    product_id = client.post("/api/products", json={"name": "Onion", "unit": "kg", "region": "South", "current_stock": 100}).get_json()["id"]
    for quantity in (10, 20, 30):
        assert client.post("/api/sales", json={"product_id": product_id, "quantity": quantity, "sale_date": "2026-09-16"}).status_code == 201
    suggestion = client.get(f"/api/sales/suggestion/{product_id}").get_json()
    assert suggestion["available"] is True
    assert suggestion["suggested_quantity"] > 0
    assert len(client.get("/api/sales").get_json()) == 3
    saved = client.post("/api/sales", json={"product_id": product_id, "quantity": 27, "entry_method": "suggested"})
    assert saved.status_code == 201
    assert saved.get_json()["quantity"] == 27
    assert len(client.get("/api/sales").get_json()) == 4


def test_future_sale_and_negative_inventory_are_rejected(client):
    client.post("/register", data=registration("dates@example.com", "Dina")); login(client, "dates@example.com")
    product_id = client.post("/api/products", json={"name": "Rice", "unit": "kg", "region": "South", "current_stock": 5}).get_json()["id"]
    future = client.post("/api/sales", json={"product_id": product_id, "quantity": 1, "sale_date": "2999-01-01"})
    assert future.status_code == 400
    oversell = client.post("/api/sales", json={"product_id": product_id, "quantity": 6})
    assert oversell.status_code == 400
    assert client.get("/api/inventory").get_json()[0]["current_stock"] == 5


def test_stock_in_and_transaction_history(client):
    client.post("/register", data=registration("stock@example.com", "Sita")); login(client, "stock@example.com")
    product_id = client.post("/api/products", json={"name": "Potato", "unit": "kg", "region": "South", "current_stock": 5}).get_json()["id"]
    response = client.post("/api/inventory/stock-in", json={"product_id": product_id, "quantity": 15})
    assert response.status_code == 201
    detail = client.get(f"/api/inventory/{product_id}").get_json()
    assert detail["current_stock"] == 20
    assert detail["transactions"][0]["transaction_type"] == "STOCK_IN"


def test_active_reorder_alert_is_updated_not_duplicated(client):
    client.post("/register", data=registration("alerts@example.com", "Alia")); login(client, "alerts@example.com")
    product_id = client.post("/api/products", json={"name": "Garlic", "unit": "kg", "region": "South", "current_stock": 2, "safety_stock": 10}).get_json()["id"]
    for _ in range(2):
        client.post("/api/inventory/stock-in", json={"product_id": product_id, "quantity": 1})
    # A sale after the stock-in operations re-evaluates the same reorder condition.
    client.post("/api/sales", json={"product_id": product_id, "quantity": 1})
    alerts = client.get("/api/alerts").get_json()
    active = [alert for alert in alerts if alert["is_read"] == 0]
    assert len(active) <= 1


def test_reorder_alert_button_and_page_link(client):
    client.post("/register", data=registration("reorder@example.com", "Rhea")); login(client, "reorder@example.com")
    product_id = client.post("/api/products", json={"name": "Jaggery", "unit": "kg", "region": "South", "current_stock": 5, "safety_stock": 10, "lead_time_days": 7}).get_json()["id"]
    response = client.get("/alerts")
    assert b"/reorder/" in response.data
    page = client.get(f"/reorder/{product_id}")
    assert page.status_code == 200
    assert b"Reorder Recommendation" in page.data
    assert b"Jaggery" in page.data


def test_notification_badge_counts_only_current_users_active_alerts(client):
    client.post("/register", data=registration("badge@example.com", "Bea")); login(client, "badge@example.com")
    first_id = client.post("/api/products", json={"name": "Jaggery", "unit": "kg", "region": "South", "current_stock": 2, "safety_stock": 10}).get_json()["id"]
    second_id = client.post("/api/products", json={"name": "Turmeric", "unit": "kg", "region": "South", "current_stock": 1, "safety_stock": 10}).get_json()["id"]
    page = client.get("/dashboard").get_data(as_text=True)
    assert 'class="notification-badge">2</span>' in page
    client.post("/api/inventory/stock-in", json={"product_id": first_id, "quantity": 20})
    page = client.get("/dashboard").get_data(as_text=True)
    assert 'class="notification-badge">1</span>' in page
    client.post("/api/inventory/stock-in", json={"product_id": second_id, "quantity": 20})
    page = client.get("/dashboard").get_data(as_text=True)
    assert "notification-badge" not in page
    assert 'href="/alerts"' in page


def test_confirm_reorder_records_request_without_changing_inventory(client):
    client.post("/register", data=registration("confirm@example.com", "Cora")); login(client, "confirm@example.com")
    product_id = client.post("/api/products", json={"name": "Rice", "unit": "kg", "region": "South", "current_stock": 8, "safety_stock": 6, "lead_time_days": 7}).get_json()["id"]
    before = client.get("/api/inventory").get_json()[0]["current_stock"]
    response = client.post(f"/reorder/{product_id}", data={"confirm": "1"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Reorder request recorded" in response.data
    after = client.get("/api/inventory").get_json()[0]["current_stock"]
    assert after == before
    assert client.get("/api/reorders").get_json()[0]["product_id"] == product_id


def test_sales_form_has_no_quantity_entry_method_toggle(client):
    client.post("/register", data=registration("salesform@example.com", "Sonia")); login(client, "salesform@example.com")
    product_id = client.post("/api/products", json={"name": "Potato", "unit": "kg", "region": "South", "current_stock": 50}).get_json()["id"]
    response = client.get("/sales/new")
    assert b"Quantity Entry Method" not in response.data
    assert b"Automatic Suggestion" not in response.data
    assert b"Suggested Quantity Based on Previous Sales" in response.data
    assert b"name=\"quantity\"" in response.data


def test_alert_lifecycle_active_alerts_and_history(client):
    client.post("/register", data=registration("alerts_lifecycle@example.com", "Lina")); login(client, "alerts_lifecycle@example.com")
    product_id = client.post("/api/products", json={"name": "Jaggery", "unit": "kg", "region": "South", "current_stock": 2, "safety_stock": 10, "lead_time_days": 7}).get_json()["id"]
    active = client.get("/api/alerts").get_json()
    assert any(item["product_id"] == product_id and item["status"] == "ACTIVE" for item in active)
    assert len([item for item in active if item["product_id"] == product_id and item["status"] == "ACTIVE"]) == 1
    client.post("/api/inventory/stock-in", json={"product_id": product_id, "quantity": 20})
    active = client.get("/api/alerts").get_json()
    assert not any(item["product_id"] == product_id and item["status"] == "ACTIVE" for item in active)
    history = [item for item in client.get("/api/alerts").get_json() if item["product_id"] == product_id and item["status"] == "RESOLVED"]
    assert len(history) >= 1
    page = client.get("/alerts").get_data(as_text=True)
    assert "ACTIVE ALERTS" in page
    assert "Alert History" in page
    assert "Jaggery" in page


def test_customer_specific_forecast_api_returns_json_with_forecast_data(client, monkeypatch):
    import app as application

    client.post("/register", data=registration("forecast_json@example.com", "Fiona")); login(client, "forecast_json@example.com")
    product_id = client.post("/api/products", json={"name": "Jaggery", "unit": "kg", "region": "South", "current_stock": 50, "safety_stock": 5, "lead_time_days": 7}).get_json()["id"]
    sales = [{"sale_date": f"2026-01-{day:02d}", "quantity": 10 + (day % 7)} for day in range(1, 60)]
    for row in sales:
        client.post("/api/sales", json={"product_id": product_id, "quantity": row["quantity"], "sale_date": row["sale_date"]})
    monkeypatch.setattr(application, "forecast_customer_lstm", lambda frame, user_id, selected_product_id, horizon: [{"Date": "2026-09-18", "Predicted Demand": 12.5 + index} for index in range(horizon)])
    response = client.post(f"/api/forecasts/{product_id}/generate", json={"horizon": 7})
    assert response.status_code == 200
    assert response.is_json is True
    payload = response.get_json()
    assert payload["model_used"] == "Customer LSTM"
    assert isinstance(payload["forecast"], list)
    assert len(payload["forecast"]) == 7
    assert all("Date" in row and "Predicted Demand" in row for row in payload["forecast"])


def test_forecast_page_contains_result_rendering_and_chart_flow(client):
    client.post("/register", data=registration("forecast_page@example.com", "Freya")); login(client, "forecast_page@example.com")
    page = client.get("/forecasts").get_data(as_text=True)
    assert 'id="forecast-result"' in page
    assert "renderResult" in page
    assert "forecast-table" in page
    assert "new Chart" in page


def test_product_import_preview_confirm_and_duplicate_protection(client):
    client.post("/register", data=registration("product_import@example.com", "Priya")); login(client, "product_import@example.com")
    content = b"Product Name,Category,Unit,Current Stock\nJaggery,Grocery,kg,50\nOnion,Vegetable,kg,-1\n"
    preview = client.post("/api/products/import", data={"file": (io.BytesIO(content), "products.csv")}, content_type="multipart/form-data")
    assert preview.status_code == 200
    assert preview.get_json()["valid"] == 1
    assert preview.get_json()["invalid"] == 1
    confirm = client.post("/api/products/import", data={"action": "confirm", "file": (io.BytesIO(content), "products.csv")}, content_type="multipart/form-data")
    assert confirm.status_code == 200
    assert confirm.get_json()["imported"] == 1
    duplicate = client.post("/api/products/import", data={"action": "confirm", "file": (io.BytesIO(b"Product Name,Category,Unit,Current Stock\nJaggery,Grocery,kg,50\n"), "products.csv")}, content_type="multipart/form-data")
    assert duplicate.get_json()["imported"] == 0
    assert duplicate.get_json()["summary"]["duplicates"] == 1


def test_sales_import_validates_products_dates_quantities_and_duplicates(client):
    client.post("/register", data=registration("sales_import@example.com", "Sana")); login(client, "sales_import@example.com")
    product_id = client.post("/api/products", json={"name": "Jaggery", "category": "Grocery", "unit": "kg", "region": "South", "current_stock": 50}).get_json()["id"]
    content = b"Date,Product,Quantity Sold,Unit\n2026-01-01,Jaggery,10,kg\n2999-01-01,Jaggery,2,kg\n2026-01-02,Unknown,1,kg\n2026-01-03,Jaggery,-1,kg\n"
    preview = client.post("/api/sales/import", data={"file": (io.BytesIO(content), "sales.csv")}, content_type="multipart/form-data")
    payload = preview.get_json()
    assert preview.status_code == 200
    assert payload["valid"] == 1
    assert payload["invalid"] == 3
    confirm = client.post("/api/sales/import", data={"action": "confirm", "file": (io.BytesIO(b"Date,Product,Quantity Sold,Unit\n2026-01-01,Jaggery,10,kg\n"), "sales.csv")}, content_type="multipart/form-data")
    assert confirm.status_code == 200
    assert confirm.get_json()["imported"] == 1
    assert client.get("/api/sales").get_json()[0]["product_id"] == product_id


def test_imports_are_user_scoped(client):
    client.post("/register", data=registration("import_a@example.com", "Asha")); login(client, "import_a@example.com")
    client.post("/api/products/import", data={"action": "confirm", "file": (io.BytesIO(b"Product Name,Category,Unit,Current Stock\nRice,Grocery,kg,20\n"), "products.csv")}, content_type="multipart/form-data")
    client.get("/logout")
    client.post("/register", data=registration("import_b@example.com", "Bina")); login(client, "import_b@example.com")
    assert client.get("/api/products").get_json() == []
    assert client.get("/api/sales").get_json() == []


def test_registration_hashes_recovery_answer_and_api_hides_hashes(client):
    payload = registration("recovery_registration@example.com", "Riya")
    assert client.post("/register", data=payload).status_code == 302
    with client.application.app_context():
        from src.database import query
        user = query("SELECT password_hash,security_question,security_answer_hash FROM users WHERE email=?", (payload["email"],), one=True)
    assert user["security_question"] == payload["security_question"]
    assert user["security_answer_hash"] != payload["security_answer"]
    assert user["password_hash"] != payload["password"]
    login(client, payload["email"])
    response = client.get("/api/me")
    assert "password_hash" not in response.get_json()
    assert "security_answer_hash" not in response.get_json()


def test_forgot_password_verification_and_single_use_reset(client):
    payload = registration("recovery_flow@example.com", "Reema")
    client.post("/register", data=payload)
    account = client.post("/api/recovery/account", json={"email": payload["email"], "phone": payload["phone"]})
    assert account.status_code == 200
    assert account.get_json()["security_question"] == payload["security_question"]
    assert client.post("/api/recovery/verify", json={"security_answer": "wrong"}).status_code == 400
    assert client.post("/api/recovery/verify", json={"security_answer": "  JAGGERY  "}).status_code == 200
    reset = client.post("/api/recovery/reset", json={"new_password": "NewSecure123", "confirm_password": "NewSecure123"})
    assert reset.status_code == 200
    assert client.post("/api/recovery/reset", json={"new_password": "AnotherSecure123", "confirm_password": "AnotherSecure123"}).status_code == 401
    assert client.post("/login", data={"email": payload["email"], "password": payload["password"]}).status_code == 200
    client.get("/logout")
    assert client.post("/login", data={"email": payload["email"], "password": "NewSecure123"}).status_code == 302


def test_recovery_account_errors_are_generic_and_rate_limited(client):
    payload = registration("recovery_limit@example.com", "Rohan")
    client.post("/register", data=payload)
    responses = [client.post("/api/recovery/account", json={"email": payload["email"], "phone": "0000000000"}) for _ in range(6)]
    assert all(response.get_json()["error"] == "The information provided could not be verified." for response in responses[:5])
    assert responses[-1].status_code == 429
    unknown = client.post("/api/recovery/account", json={"email": "missing@example.com", "phone": payload["phone"]})
    assert unknown.get_json()["error"] == "Too many attempts. Please try again later." or unknown.get_json()["error"] == "The information provided could not be verified."