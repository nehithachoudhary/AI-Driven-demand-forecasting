import json
import hashlib
import re
import secrets
from datetime import date, datetime, timedelta
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, Response, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from config import DATA_PATH, DEFAULT_FORECAST_HORIZON, DEFAULT_LEAD_TIME, DEFAULT_SERVICE_LEVEL, METRICS_DIR, MODELS_DIR
from src.database import close_db, execute, get_db, init_db, query
from src.imports import read_import_file, validate_products, validate_sales
from src.customer_forecasting import MIN_CUSTOMER_RECORDS, customer_model_status, forecast_customer_lstm, train_customer_lstm
from src.forecasting import recursive_lstm_forecast, recursive_tree_forecast
from src.inventory import inventory_recommendation
from src.preprocessing import load_and_validate

app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = "local-development-secret-change-me"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
SECURITY_QUESTIONS = (
    "What was the name of your first school?",
    "What was your first wholesale product?",
    "What is your favorite product?",
    "What was the name of your first workplace?",
    "What is your favorite place?",
)
RECOVERY_GENERIC_ERROR = "The information provided could not be verified."
RECOVERY_WINDOW = timedelta(minutes=15)
RECOVERY_BLOCK = timedelta(minutes=15)
RECOVERY_MAX_FAILURES = 5
RECOVERY_TOKEN_LIFETIME = timedelta(minutes=10)
DATA, _ = load_and_validate(DATA_PATH)
METRICS = pd.read_csv(METRICS_DIR / "model_comparison.csv") if (METRICS_DIR / "model_comparison.csv").exists() else pd.DataFrame()
BEST_MODEL = str(METRICS.iloc[0]["Model"]) if not METRICS.empty else "Random Forest"
MODELS = {}
for name, filename in (("Random Forest", "random_forest.joblib"), ("XGBoost", "xgboost.joblib")):
    path = MODELS_DIR / filename
    if path.exists(): MODELS[name] = joblib.load(path)
LSTM_MODEL = None
LSTM_SCALER = None
if (MODELS_DIR / "lstm.keras").exists() and (MODELS_DIR / "lstm_scaler.joblib").exists():
    import tensorflow as tf
    LSTM_MODEL = tf.keras.models.load_model(MODELS_DIR / "lstm.keras")
    LSTM_SCALER = joblib.load(MODELS_DIR / "lstm_scaler.joblib")


@app.before_request
def load_user():
    user_id = session.get("user_id")
    g.user = query("SELECT * FROM users WHERE id=?", (user_id,), one=True) if user_id else None
    g.active_alert_count = 0
    if g.user:
        sync_user_reorder_alerts()
        g.active_alert_count = query("SELECT COUNT(*) total FROM alerts WHERE user_id=? AND status='ACTIVE'", (g.user["id"],), one=True)["total"]


@app.teardown_appcontext
def teardown_db(error): close_db(error)


with app.app_context(): init_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith("/api/"): return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def error(message, status=400): return jsonify({"success": False, "error": message}), status


@app.after_request
def normalize_api_responses(response):
    if request.path.startswith("/api/") and response.mimetype == "application/json":
        payload = response.get_json(silent=True)
        if isinstance(payload, dict) and "success" not in payload:
            payload["success"] = response.status_code < 400
            if response.status_code >= 400 and "error" not in payload:
                payload["error"] = response.status or "Request failed"
            response.set_data(json.dumps(payload))
            response.content_type = "application/json"
            response.headers.pop("Content-Length", None)
    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if request.path.startswith("/api/"):
        app.logger.warning("API %s %s failed with %s", request.method, request.path, error.code)
        return jsonify({"success": False, "error": error.description or error.name}), error.code
    return error


@app.errorhandler(Exception)
def handle_unexpected_api_error(error):
    if request.path.startswith("/api/"):
        app.logger.exception("Unhandled API exception for %s %s", request.method, request.path)
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500
    raise error


def valid_email(value): return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value or ""))


def valid_phone(value): return bool(re.fullmatch(r"[0-9+()\-\s]{7,20}", value or ""))


def normalize_recovery_answer(value): return " ".join(str(value or "").strip().casefold().split())


def utc_now(): return datetime.utcnow()


def recovery_key(purpose): return f"{purpose}:{request.remote_addr or 'unknown'}"


def recovery_allowed(purpose):
    row = query("SELECT * FROM recovery_attempts WHERE attempt_key=?", (recovery_key(purpose),), one=True)
    now = utc_now()
    if row and row["blocked_until"] and datetime.fromisoformat(row["blocked_until"]) > now:
        return False
    if row and datetime.fromisoformat(row["window_started_at"]) + RECOVERY_WINDOW <= now:
        execute("DELETE FROM recovery_attempts WHERE attempt_key=?", (recovery_key(purpose),))
    return True


def record_recovery_failure(purpose):
    key = recovery_key(purpose); now = utc_now()
    row = query("SELECT * FROM recovery_attempts WHERE attempt_key=?", (key,), one=True)
    if not row or datetime.fromisoformat(row["window_started_at"]) + RECOVERY_WINDOW <= now:
        execute("INSERT OR REPLACE INTO recovery_attempts (attempt_key,failure_count,window_started_at,blocked_until) VALUES (?,?,?,NULL)", (key, 1, now.isoformat()))
        return
    failures = row["failure_count"] + 1
    blocked = (now + RECOVERY_BLOCK).isoformat() if failures >= RECOVERY_MAX_FAILURES else None
    execute("UPDATE recovery_attempts SET failure_count=?,blocked_until=? WHERE attempt_key=?", (failures, blocked, key))


def clear_recovery_failures(purpose): execute("DELETE FROM recovery_attempts WHERE attempt_key=?", (recovery_key(purpose),))


def issue_recovery_authorization(user_id):
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = (utc_now() + RECOVERY_TOKEN_LIFETIME).isoformat()
    execute("UPDATE recovery_tokens SET used_at=? WHERE user_id=? AND used_at IS NULL", (utc_now().isoformat(), user_id))
    cursor = execute("INSERT INTO recovery_tokens (user_id,token_hash,expires_at) VALUES (?,?,?)", (user_id, token_hash, expires_at))
    session["recovery_token_id"] = cursor.lastrowid
    return raw_token


def recovery_token_user():
    token_id = session.get("recovery_token_id")
    if not token_id: return None
    token = query("SELECT * FROM recovery_tokens WHERE id=? AND used_at IS NULL", (token_id,), one=True)
    if not token or datetime.fromisoformat(token["expires_at"]) <= utc_now():
        session.pop("recovery_token_id", None)
        return None
    return token


def validate_past_or_today(value, label="sale"):
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {label} date")
    if parsed > date.today():
        raise ValueError("Sales can only be recorded for today or a previous date.")
    return parsed.isoformat()


def parse_product_form(payload, current_stock_default=0):
    if any(not str(payload.get(field, "")).strip() for field in ("name", "unit", "region")): raise ValueError("Product name, unit, and region are required")
    try:
        stock = float(payload.get("current_stock", current_stock_default)); safety = float(payload.get("safety_stock", 0)); lead = int(payload.get("lead_time_days", DEFAULT_LEAD_TIME))
    except (TypeError, ValueError): raise ValueError("Stock, safety stock, and lead time must be numeric")
    if stock < 0 or safety < 0 or lead < 0: raise ValueError("Stock, safety stock, and lead time cannot be negative")
    return {"name": str(payload["name"]).strip(), "category": str(payload.get("category", "General")).strip(), "unit": str(payload["unit"]).strip(), "region": str(payload["region"]).strip(), "current_stock": stock, "safety_stock": safety, "lead_time_days": lead, "description": str(payload.get("description", "")).strip()}


def user_product(product_id):
    product = query("SELECT * FROM products WHERE id=? AND user_id=? AND is_active=1", (product_id, g.user["id"]), one=True)
    if product is None: raise ValueError("Product was not found or you do not have permission to access it")
    return product


def inventory_status_for_stock(current_stock, reorder_point, safety_stock):
    if current_stock <= reorder_point:
        return "REORDER REQUIRED"
    if current_stock <= safety_stock:
        return "LOW STOCK"
    return "HEALTHY STOCK"


def inventory_row(product):
    recent = query("SELECT quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date >= date('now','-30 day')", (g.user["id"], product["id"]))
    quantities = [float(row["quantity"]) for row in recent]
    average_daily = sum(quantities) / 30 if quantities else 0.0
    demand_std = pd.Series(quantities).std() if len(quantities) > 1 else 0.0
    calculation = inventory_recommendation(average_daily * max(product["lead_time_days"], 1), product["current_stock"], float(demand_std or 0), product["lead_time_days"] or 1, DEFAULT_SERVICE_LEVEL)
    calculation["reorder_point"] = max(calculation["reorder_point"], float(product["safety_stock"]))
    calculation["suggested_reorder_quantity"] = max(0.0, average_daily * max(product["lead_time_days"], 1) + float(product["safety_stock"]) - float(product["current_stock"]))
    coverage = product["current_stock"] / average_daily if average_daily > 0 else None
    return {**dict(product), "average_daily_demand": average_daily, "reorder_point": calculation["reorder_point"], "suggested_order_quantity": calculation["suggested_reorder_quantity"], "stock_coverage": coverage, "status": inventory_status_for_stock(product["current_stock"], calculation["reorder_point"], float(product["safety_stock"]))}


def sales_suggestion(product_id):
    today = date.today().isoformat()
    rows = query("SELECT quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date<=? ORDER BY sale_date DESC, id DESC LIMIT 7", (g.user["id"], product_id, today))
    if len(rows) < 3:
        return None
    quantities = [float(row["quantity"]) for row in reversed(rows)]
    weights = list(range(1, len(quantities) + 1))
    return sum(quantity * weight for quantity, weight in zip(quantities, weights)) / sum(weights)


def sync_reorder_alert(product, calculation):
    needs_alert = product["current_stock"] <= calculation["reorder_point"]
    existing = query("SELECT * FROM alerts WHERE user_id=? AND product_id=? AND alert_type='REORDER' ORDER BY created_at DESC, id DESC", (g.user["id"], product["id"]))
    active = next((row for row in existing if row["status"] == "ACTIVE"), None)
    if needs_alert:
        if active:
            if float(active["previous_stock"]) != float(product["current_stock"]):
                execute("UPDATE alerts SET message=?,severity=?,previous_stock=?,status='ACTIVE',resolved_at=NULL WHERE id=?", ("Reorder recommended", "critical", float(product["current_stock"]), active["id"]))
        else:
            execute("INSERT INTO alerts (user_id,product_id,alert_type,message,severity,previous_stock,status) VALUES (?,?,?,?,?,?,?)", (g.user["id"], product["id"], "REORDER", "Reorder recommended", "critical", float(product["current_stock"]), "ACTIVE"))
        for row in existing:
            if row["id"] != (active["id"] if active else None):
                execute("UPDATE alerts SET status='RESOLVED', resolved_at=CURRENT_TIMESTAMP, is_read=1 WHERE id=? AND user_id=?", (row["id"], g.user["id"]))
    else:
        for row in existing:
            if row["status"] != "RESOLVED":
                execute("UPDATE alerts SET status='RESOLVED', resolved_at=CURRENT_TIMESTAMP, is_read=1 WHERE id=? AND user_id=?", (row["id"], g.user["id"]))


def sync_user_reorder_alerts():
    for product in query("SELECT * FROM products WHERE user_id=? AND is_active=1", (g.user["id"],)):
        sync_reorder_alert(product, inventory_row(product))


def reorder_summary(product):
    product_row = inventory_row(product)
    sales = query("SELECT sale_date, quantity, unit, buyer, selling_price FROM sales WHERE user_id=? AND product_id=? ORDER BY sale_date ASC, id ASC", (g.user["id"], product["id"]))
    quantities = [float(row["quantity"]) for row in sales]
    total_sales = sum(quantities)
    if sales:
        start = pd.to_datetime(sales[0]["sale_date"])
        end = pd.to_datetime(sales[-1]["sale_date"])
        days = max((end - start).days + 1, 1)
    else:
        days = 1
    average_daily = total_sales / days if total_sales else 0.0
    calculation = inventory_recommendation(average_daily * max(product["lead_time_days"], 1), product_row["current_stock"], float(pd.Series(quantities).std() if len(quantities) > 1 else 0.0), product["lead_time_days"] or 1, DEFAULT_SERVICE_LEVEL)
    calculation["reorder_point"] = max(calculation["reorder_point"], float(product["safety_stock"]))
    calculation["target_stock"] = max(0.0, float(calculation["forecast_demand"]) + float(calculation["safety_stock"]))
    calculation["recommended_order_quantity"] = max(0.0, calculation["target_stock"] - float(product_row["current_stock"]))
    forecast = None
    model_status = customer_model_status(pd.DataFrame([{"sale_date": row["sale_date"], "quantity": row["quantity"]} for row in sales], columns=("sale_date", "quantity")), g.user["id"], product["id"])
    if model_status["status"] == "trained":
        frame = pd.DataFrame([{"sale_date": row["sale_date"], "quantity": row["quantity"]} for row in sales], columns=("sale_date", "quantity"))
        try:
            forecast = forecast_customer_lstm(frame, g.user["id"], product["id"], 7)
        except ValueError:
            forecast = None
    return {"product": product_row, "sales": sales, "summary": {"total_sales": total_sales, "average_daily_sales": average_daily, "highest_daily_sales": max(quantities) if quantities else 0.0, "lowest_daily_sales": min(quantities) if quantities else 0.0, "record_count": len(sales), "first_recorded_sale": sales[0]["sale_date"] if sales else None, "latest_recorded_sale": sales[-1]["sale_date"] if sales else None}, "calculation": calculation, "forecast": forecast, "model_status": model_status}


@app.get("/")
def index(): return redirect(url_for("dashboard")) if g.user else redirect(url_for("login"))


@app.route("/register", methods=("GET", "POST"))
def register():
    if request.method == "POST":
        p = request.form; required = ("full_name", "business_name", "email", "phone", "address", "city", "state", "region", "security_question", "security_answer", "password", "confirm_password")
        if any(not p.get(field, "").strip() for field in required): flash("All registration fields are required.", "error")
        elif not valid_email(p.get("email")): flash("Enter a valid email address.", "error")
        elif not valid_phone(p.get("phone")): flash("Enter a valid mobile number.", "error")
        elif p.get("security_question") not in SECURITY_QUESTIONS: flash("Select a valid security question.", "error")
        elif not normalize_recovery_answer(p.get("security_answer")): flash("Security answer is required.", "error")
        elif len(p.get("password", "")) < 8: flash("Password must be at least 8 characters.", "error")
        elif p.get("password") != p.get("confirm_password"): flash("Passwords do not match.", "error")
        elif query("SELECT id FROM users WHERE email=?", (p["email"].lower().strip(),), one=True): flash("That email is already registered.", "error")
        else:
            cursor = execute("INSERT INTO users (full_name,business_name,email,phone,address,city,state,region,password_hash,security_question,security_answer_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (p["full_name"].strip(), p["business_name"].strip(), p["email"].lower().strip(), p["phone"].strip(), p["address"].strip(), p["city"].strip(), p["state"].strip(), p["region"].strip(), generate_password_hash(p["password"]), p["security_question"], generate_password_hash(normalize_recovery_answer(p["security_answer"]))))
            execute("INSERT INTO user_settings (user_id,default_region) VALUES (?,?)", (cursor.lastrowid, p["region"].strip()))
            flash("Account created. Please log in.", "success"); return redirect(url_for("login"))
    return render_template("register.html", form=request.form if request.method == "POST" else {}, security_questions=SECURITY_QUESTIONS)


@app.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        user = query("SELECT * FROM users WHERE email=?", (request.form.get("email", "").lower().strip(),), one=True)
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear(); session["user_id"] = user["id"]; return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.get("/forgot-password")
def forgot_password():
    session.pop("recovery_token_id", None)
    return render_template("forgot_password.html", security_questions=SECURITY_QUESTIONS)


@app.get("/reset-password")
def reset_password_page():
    if recovery_token_user() is None:
        flash(RECOVERY_GENERIC_ERROR, "error")
        return redirect(url_for("forgot_password"))
    return render_template("reset_password.html")


@app.get("/logout")
def logout(): session.clear(); return redirect(url_for("login"))


@app.get("/dashboard")
@login_required
def dashboard():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    sales_today = query("SELECT COALESCE(SUM(quantity),0) total FROM sales WHERE user_id=? AND sale_date=date('now')", (g.user["id"],), one=True)["total"]
    alerts = g.active_alert_count
    return render_template("dashboard.html", products=[inventory_row(p) for p in products], sales_today=sales_today, alerts=alerts, page_title="Dashboard")


@app.get("/products")
@login_required
def products_page():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    return render_template("products.html", products=[inventory_row(p) for p in products], page_title="My Products")


@app.get("/products/import")
@login_required
def products_import_page():
    return render_template("import_data.html", import_type="products", page_title="Import Products")


@app.get("/products/import/template")
@login_required
def products_import_template():
    return Response("Product Name,Category,Unit,Current Stock\n", mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=products_template.csv"})


@app.route("/products/new", methods=("GET", "POST"))
@login_required
def product_new():
    if request.method == "POST":
        try:
            v = parse_product_form(request.form)
            cursor = execute("INSERT INTO products (user_id,name,category,unit,region,current_stock,safety_stock,lead_time_days,description) VALUES (?,?,?,?,?,?,?,?,?)", (g.user["id"], v["name"], v["category"], v["unit"], v["region"], v["current_stock"], v["safety_stock"], v["lead_time_days"], v["description"]))
            product = {"id": cursor.lastrowid, "user_id": g.user["id"], **v}
            sync_reorder_alert(product, inventory_row(product))
            flash("Product saved.", "success"); return redirect(url_for("products_page"))
        except ValueError as exc: flash(str(exc), "error")
    return render_template("product_form.html", product=None, page_title="Add Product")


@app.route("/products/<int:product_id>/edit", methods=("GET", "POST"))
@login_required
def product_edit(product_id):
    try: product = user_product(product_id)
    except ValueError as exc: flash(str(exc), "error"); return redirect(url_for("products_page"))
    if request.method == "POST":
        try:
            v = parse_product_form(request.form, product["current_stock"])
            execute("UPDATE products SET name=?,category=?,unit=?,region=?,safety_stock=?,lead_time_days=?,description=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (v["name"], v["category"], v["unit"], v["region"], v["safety_stock"], v["lead_time_days"], v["description"], product_id, g.user["id"]))
            sync_reorder_alert({**dict(product), **v}, inventory_row({**dict(product), **v}))
            flash("Product updated.", "success"); return redirect(url_for("products_page"))
        except ValueError as exc: flash(str(exc), "error")
    return render_template("product_form.html", product=product, page_title="Edit Product")


@app.post("/products/<int:product_id>/delete")
@login_required
def product_delete(product_id):
    try: user_product(product_id)
    except ValueError as exc: flash(str(exc), "error"); return redirect(url_for("products_page"))
    execute("UPDATE products SET is_active=0,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (product_id, g.user["id"])); flash("Product archived.", "success"); return redirect(url_for("products_page"))


@app.get("/sales")
@login_required
def sales_page():
    sales = query("SELECT s.*,p.name product_name FROM sales s JOIN products p ON p.id=s.product_id WHERE s.user_id=? ORDER BY s.sale_date DESC,s.id DESC", (g.user["id"],)); products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    return render_template("sales.html", sales=sales, products=products, page_title="Sales")


@app.get("/sales/import")
@login_required
def sales_import_page():
    return render_template("import_data.html", import_type="sales", page_title="Import Sales")


@app.get("/sales/import/template")
@login_required
def sales_import_template():
    return Response("Date,Product,Quantity Sold,Unit,Amount,Buyer\n", mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=sales_template.csv"})


@app.route("/sales/new", methods=("GET", "POST"))
@login_required
def sale_new():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    if request.method == "POST":
        try:
            product = user_product(int(request.form["product_id"])); quantity = float(request.form["quantity"])
            if quantity <= 0: raise ValueError("Quantity must be positive")
            if quantity > product["current_stock"]: raise ValueError("Sale quantity exceeds available stock")
            sale_date = request.form.get("sale_date") or date.today().isoformat()
            sale_date = validate_past_or_today(sale_date)
            entry_method = "Suggested + Confirmed" if request.form.get("entry_method") == "suggested" else "Manual"
            cursor = execute("INSERT INTO sales (user_id,product_id,sale_date,quantity,unit,selling_price,buyer,notes,entry_method) VALUES (?,?,?,?,?,?,?,?,?)", (g.user["id"], product["id"], sale_date, quantity, product["unit"], float(request.form["selling_price"]) if request.form.get("selling_price") else None, request.form.get("buyer", ""), request.form.get("notes", ""), entry_method))
            new_stock = product["current_stock"] - quantity; execute("UPDATE products SET current_stock=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (new_stock, product["id"], g.user["id"])); execute("INSERT INTO inventory_transactions (user_id,product_id,transaction_type,quantity,reference_id,notes) VALUES (?,?,?,?,?,?)", (g.user["id"], product["id"], "SALE", -quantity, cursor.lastrowid, "Recorded sale"))
            calculation = inventory_row({**dict(product), "current_stock": new_stock})
            sync_reorder_alert({**dict(product), "current_stock": new_stock}, calculation)
            flash("Sale recorded and inventory updated.", "success"); return redirect(url_for("sales_page"))
        except (ValueError, TypeError, KeyError) as exc: flash(str(exc), "error")
    return render_template("sale_form.html", products=products, page_title="Record Sale", now_date=date.today().isoformat())


@app.get("/inventory")
@login_required
def inventory_page():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],)); return render_template("inventory.html", products=[inventory_row(p) for p in products], page_title="Inventory")


@app.get("/alerts")
@login_required
def alerts_page():
    sync_user_reorder_alerts()
    active_alerts = query("SELECT a.*,p.name product_name,p.current_stock,p.safety_stock,p.lead_time_days FROM alerts a JOIN products p ON p.id=a.product_id WHERE a.user_id=? AND a.status='ACTIVE' ORDER BY a.created_at DESC", (g.user["id"],))
    history_alerts = query("SELECT a.*,p.name product_name,p.current_stock,p.safety_stock,p.lead_time_days FROM alerts a JOIN products p ON p.id=a.product_id WHERE a.user_id=? AND a.status='RESOLVED' ORDER BY a.resolved_at DESC, a.created_at DESC", (g.user["id"],))
    return render_template("alerts.html", active_alerts=active_alerts, history_alerts=history_alerts, page_title="Alerts")


@app.get("/reorder/<int:product_id>")
@login_required
def reorder_page(product_id):
    try:
        product = user_product(product_id)
    except ValueError as exc:
        flash(str(exc), "error"); return redirect(url_for("alerts_page"))
    payload = reorder_summary(product)
    if payload["product"]["current_stock"] > payload["calculation"]["reorder_point"]:
        flash("This product does not currently require a reorder.", "info"); return redirect(url_for("inventory_page"))
    return render_template("reorder.html", payload=payload, page_title="Reorder Recommendation")


@app.post("/reorder/<int:product_id>")
@login_required
def reorder_submit(product_id):
    try:
        product = user_product(product_id)
    except ValueError as exc:
        flash(str(exc), "error"); return redirect(url_for("alerts_page"))
    payload = reorder_summary(product)
    recommendation = payload["calculation"]["recommended_order_quantity"]
    existing = query("SELECT id FROM reorder_requests WHERE user_id=? AND product_id=? AND status='PENDING' ORDER BY created_at DESC LIMIT 1", (g.user["id"], product_id), one=True)
    if existing is None:
        execute("INSERT INTO reorder_requests (user_id,product_id,product_name,recommended_quantity,confirmed_quantity,current_stock,safety_stock,reorder_point,target_stock,status) VALUES (?,?,?,?,?,?,?,?,?,?)", (g.user["id"], product["id"], product["name"], recommendation, recommendation, payload["product"]["current_stock"], float(product["safety_stock"]), payload["calculation"]["reorder_point"], payload["calculation"]["target_stock"], "PENDING"))
    flash("Reorder request recorded. Inventory will update only when stock is received.", "success")
    return redirect(url_for("alerts_page"))


@app.get("/api/reorders")
@login_required
def api_reorders():
    return jsonify([dict(row) for row in query("SELECT * FROM reorder_requests WHERE user_id=? ORDER BY created_at DESC", (g.user["id"],))])


@app.route("/profile", methods=("GET", "POST"))
@login_required
def profile():
    if request.method == "POST":
        values = (request.form.get("full_name", "").strip(), request.form.get("phone", "").strip(), request.form.get("business_name", "").strip(), request.form.get("address", "").strip(), request.form.get("city", "").strip(), request.form.get("state", "").strip(), request.form.get("region", "").strip())
        if not all(values) or not valid_phone(values[1]): flash("Complete the profile fields with a valid mobile number.", "error")
        elif request.form.get("security_question") and not normalize_recovery_answer(request.form.get("security_answer")): flash("Security answer is required when updating recovery information.", "error")
        elif request.form.get("security_question") not in ("", *SECURITY_QUESTIONS): flash("Select a valid security question.", "error")
        else:
            if request.form.get("security_question"):
                execute("UPDATE users SET full_name=?,phone=?,business_name=?,address=?,city=?,state=?,region=?,security_question=?,security_answer_hash=? WHERE id=?", (*values, request.form["security_question"], generate_password_hash(normalize_recovery_answer(request.form["security_answer"])), g.user["id"]))
            else:
                execute("UPDATE users SET full_name=?,phone=?,business_name=?,address=?,city=?,state=?,region=? WHERE id=?", (*values, g.user["id"]))
            flash("Profile updated.", "success"); return redirect(url_for("profile"))
    return render_template("profile.html", page_title="Profile", security_questions=SECURITY_QUESTIONS)


@app.route("/change-password", methods=("GET", "POST"))
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not check_password_hash(g.user["password_hash"], current_password):
            flash("Current password is incorrect.", "error")
        elif len(new_password) < 8:
            flash("New password must be at least 8 characters.", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "error")
        elif check_password_hash(g.user["password_hash"], new_password):
            flash("New password must be different from your current password.", "error")
        else:
            execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new_password), g.user["id"]))
            flash("Password updated successfully.", "success")
            return redirect(url_for("settings"))
    return render_template("change_password.html", page_title="Change Password")


@app.route("/settings", methods=("GET", "POST"))
@login_required
def settings():
    current = query("SELECT * FROM user_settings WHERE user_id=?", (g.user["id"],), one=True)
    if request.method == "POST":
        try: horizon = int(request.form.get("default_forecast_horizon", 7))
        except ValueError: horizon = 7
        if horizon not in (7, 14, 30): flash("Forecast horizon must be 7, 14, or 30 days.", "error")
        else:
            execute("UPDATE user_settings SET default_forecast_horizon=?,default_region=?,default_unit=?,alert_enabled=? WHERE user_id=?", (horizon, request.form.get("default_region", "").strip(), request.form.get("default_unit", "").strip(), 1 if request.form.get("alert_enabled") else 0, g.user["id"])); flash("Settings updated.", "success"); current = query("SELECT * FROM user_settings WHERE user_id=?", (g.user["id"],), one=True)
    return render_template("settings.html", settings=current, page_title="Settings")


@app.get("/reports")
@login_required
def reports(): return render_template("reports.html", page_title="Reports")


@app.route("/forecasts", methods=("GET", "POST"))
@login_required
def forecasts_page():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    result = None
    selected = request.form.get("product_id") if request.method == "POST" else None
    status = None
    if selected:
        try:
            product = user_product(int(selected))
            sales = query("SELECT sale_date,quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date<=date('now') ORDER BY sale_date", (g.user["id"], product["id"]))
            status = customer_model_status(pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity")), g.user["id"], product["id"])
            if request.method == "POST":
                if request.form.get("action") == "train":
                    metadata = train_customer_lstm(pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity")), g.user["id"], product["id"])
                    result = {"success": f"Customer LSTM trained on {metadata['training_record_count']} actual sales records."}
                    status = customer_model_status(pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity")), g.user["id"], product["id"])
                elif request.form.get("action") == "forecast":
                    result = {"forecast": forecast_customer_lstm(pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity")), g.user["id"], product["id"], int(request.form.get("horizon", 7)))}
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc), "detail": ""}
    return render_template("forecasts.html", products=products, result=result, status=status, selected=selected, min_records=MIN_CUSTOMER_RECORDS, page_title="Forecasts")


@app.get("/api/me")
@login_required
def api_me():
    return jsonify({field: g.user[field] for field in ("id", "full_name", "business_name", "email", "phone", "address", "city", "state", "region")})


@app.post("/api/register")
def api_register():
    payload = request.get_json(silent=True) or {}
    required = ("full_name", "business_name", "email", "phone", "address", "city", "state", "region", "security_question", "security_answer", "password", "confirm_password")
    if any(not str(payload.get(field, "")).strip() for field in required): return error("All registration fields are required")
    if not valid_email(payload["email"]): return error("Enter a valid email address")
    if not valid_phone(payload["phone"]): return error("Enter a valid mobile number")
    if payload["security_question"] not in SECURITY_QUESTIONS: return error("Select a valid security question")
    if not normalize_recovery_answer(payload["security_answer"]): return error("Security answer is required")
    if len(payload["password"]) < 8: return error("Password must be at least 8 characters")
    if payload["password"] != payload["confirm_password"]: return error("Passwords do not match")
    email = payload["email"].lower().strip()
    if query("SELECT id FROM users WHERE email=?", (email,), one=True): return error("That email is already registered", 409)
    cursor = execute("INSERT INTO users (full_name,business_name,email,phone,address,city,state,region,password_hash,security_question,security_answer_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (payload["full_name"].strip(), payload["business_name"].strip(), email, payload["phone"].strip(), payload["address"].strip(), payload["city"].strip(), payload["state"].strip(), payload["region"].strip(), generate_password_hash(payload["password"]), payload["security_question"], generate_password_hash(normalize_recovery_answer(payload["security_answer"]))))
    execute("INSERT INTO user_settings (user_id,default_region) VALUES (?,?)", (cursor.lastrowid, payload["region"].strip()))
    return jsonify({"id": cursor.lastrowid, "email": email}), 201


@app.post("/api/recovery/account")
def api_recovery_account():
    if not recovery_allowed("account"): return error("Too many attempts. Please try again later.", 429)
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).lower().strip(); phone = str(payload.get("phone", "")).strip()
    user = query("SELECT id,security_question,security_answer_hash FROM users WHERE email=? AND phone=?", (email, phone), one=True)
    if not user or not user["security_question"] or not user["security_answer_hash"]:
        record_recovery_failure("account")
        return error(RECOVERY_GENERIC_ERROR, 400)
    clear_recovery_failures("account")
    session["recovery_user_id"] = user["id"]
    return jsonify({"success": True, "security_question": user["security_question"]})


@app.post("/api/recovery/verify")
def api_recovery_verify():
    if not recovery_allowed("answer"): return error("Too many attempts. Please try again later.", 429)
    user_id = session.get("recovery_user_id"); payload = request.get_json(silent=True) or {}
    user = query("SELECT id,security_answer_hash FROM users WHERE id=?", (user_id,), one=True) if user_id else None
    if not user or not user["security_answer_hash"] or not check_password_hash(user["security_answer_hash"], normalize_recovery_answer(payload.get("security_answer"))):
        record_recovery_failure("answer")
        return error(RECOVERY_GENERIC_ERROR, 400)
    clear_recovery_failures("answer")
    issue_recovery_authorization(user["id"])
    session.pop("recovery_user_id", None)
    return jsonify({"success": True})


@app.post("/api/recovery/reset")
def api_recovery_reset():
    if not recovery_allowed("reset"): return error("Too many attempts. Please try again later.", 429)
    token = recovery_token_user(); payload = request.get_json(silent=True) or {}
    new_password = str(payload.get("new_password", "")); confirm_password = str(payload.get("confirm_password", ""))
    if token is None: return error(RECOVERY_GENERIC_ERROR, 401)
    if len(new_password) < 8: return error("Password must be at least 8 characters")
    if new_password != confirm_password: return error("Passwords do not match")
    user = query("SELECT id,password_hash FROM users WHERE id=?", (token["user_id"],), one=True)
    if check_password_hash(user["password_hash"], new_password): return error("New password must be different from your current password")
    execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new_password), user["id"]))
    execute("UPDATE recovery_tokens SET used_at=? WHERE id=?", (utc_now().isoformat(), token["id"]))
    session.clear()
    return jsonify({"success": True, "message": "Password updated successfully."})


@app.post("/api/login")
def api_login():
    payload = request.get_json(silent=True) or {}
    user = query("SELECT * FROM users WHERE email=?", (str(payload.get("email", "")).lower().strip(),), one=True)
    if not user or not check_password_hash(user["password_hash"], payload.get("password", "")): return error("Invalid email or password", 401)
    session.clear(); session["user_id"] = user["id"]
    return jsonify({"id": user["id"], "email": user["email"], "full_name": user["full_name"]})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"logged_out": True})


@app.route("/api/products", methods=("GET", "POST"))
@login_required
def api_products():
    if request.method == "GET": return jsonify([dict(row) for row in query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))])
    try:
        v = parse_product_form(request.get_json(silent=True) or {})
        cursor = execute("INSERT INTO products (user_id,name,category,unit,region,current_stock,safety_stock,lead_time_days,description) VALUES (?,?,?,?,?,?,?,?,?)", (g.user["id"], v["name"], v["category"], v["unit"], v["region"], v["current_stock"], v["safety_stock"], v["lead_time_days"], v["description"]))
        product = {"id": cursor.lastrowid, "user_id": g.user["id"], **v}
        sync_reorder_alert(product, inventory_row(product))
        return jsonify({"id": cursor.lastrowid, **v}), 201
    except ValueError as exc: return error(str(exc))


def _import_summary(rows):
    valid = [row for row in rows if not row["errors"]]
    invalid = [row for row in rows if row["errors"]]
    duplicates = [row for row in rows if any("Duplicate" in message or "already exists" in message for message in row["errors"])]
    return {"rows_detected": len(rows), "valid": len(valid), "invalid": len(invalid), "duplicates": len(duplicates), "rows": rows}


@app.post("/api/products/import")
@login_required
def api_import_products():
    upload = request.files.get("file")
    if upload is None:
        return error("Select a CSV or XLSX file")
    try:
        filename, frame = read_import_file(upload)
        existing_names = {str(row["name"]).strip().casefold() for row in query("SELECT name FROM products WHERE user_id=? AND is_active=1", (g.user["id"],))}
        rows = validate_products(frame, g.user["region"], existing_names)
        summary = _import_summary(rows)
        summary["filename"] = filename
        if request.form.get("action", "preview") != "confirm":
            return jsonify({"success": True, "preview": True, **summary})
        valid_rows = [row for row in rows if not row["errors"]]
        if not valid_rows:
            return jsonify({"success": True, "imported": 0, "skipped": len(rows), "summary": summary})
        connection = get_db()
        try:
            connection.executemany("INSERT INTO products (user_id,name,category,unit,region,current_stock,safety_stock,lead_time_days,description) VALUES (?,?,?,?,?,?,?,?,?)", [(g.user["id"], row["name"], row["category"], row["unit"], row["region"], row["current_stock"], row["safety_stock"], row["lead_time_days"], "Imported from " + filename) for row in valid_rows])
            connection.commit()
        except Exception:
            connection.rollback()
            app.logger.exception("Product import failed")
            return error("Products could not be imported", 500)
        for row in query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY id DESC LIMIT ?", (g.user["id"], len(valid_rows))):
            sync_reorder_alert(row, inventory_row(row))
        return jsonify({"success": True, "imported": len(valid_rows), "skipped": len(rows) - len(valid_rows), "summary": summary})
    except ValueError as exc:
        return error(str(exc), 422)


@app.route("/api/products/<int:product_id>", methods=("PUT", "DELETE"))
@login_required
def api_product(product_id):
    try: product = user_product(product_id)
    except ValueError as exc: return error(str(exc), 404)
    if request.method == "DELETE": execute("UPDATE products SET is_active=0 WHERE id=? AND user_id=?", (product_id, g.user["id"])); return jsonify({"deleted": True})
    try:
        v = parse_product_form(request.get_json(silent=True) or {}, product["current_stock"])
        execute("UPDATE products SET name=?,category=?,unit=?,region=?,safety_stock=?,lead_time_days=?,description=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (v["name"], v["category"], v["unit"], v["region"], v["safety_stock"], v["lead_time_days"], v["description"], product_id, g.user["id"]))
        sync_reorder_alert({**dict(product), **v}, inventory_row({**dict(product), **v}))
        return jsonify({"id": product_id, **v})
    except ValueError as exc: return error(str(exc))


@app.route("/api/sales", methods=("GET", "POST"))
@login_required
def api_sales():
    if request.method == "GET": return jsonify([dict(row) for row in query("SELECT s.*,p.name product_name FROM sales s JOIN products p ON p.id=s.product_id WHERE s.user_id=? ORDER BY s.sale_date DESC", (g.user["id"],))])
    payload = request.get_json(silent=True) or {}
    try:
        product = user_product(int(payload["product_id"])); quantity = float(payload["quantity"])
        if quantity <= 0: raise ValueError("Quantity must be positive")
        if quantity > product["current_stock"]: raise ValueError(f"Insufficient stock. Available stock: {product['current_stock']} {product['unit']}.")
        sale_date = payload.get("sale_date", date.today().isoformat())
        sale_date = validate_past_or_today(sale_date)
        entry_method = "Suggested + Confirmed" if payload.get("entry_method") == "suggested" else "Manual"
        cursor = execute("INSERT INTO sales (user_id,product_id,sale_date,quantity,unit,selling_price,buyer,notes,entry_method) VALUES (?,?,?,?,?,?,?,?,?)", (g.user["id"], product["id"], sale_date, quantity, product["unit"], payload.get("selling_price"), payload.get("buyer", ""), payload.get("notes", ""), entry_method))
        new_stock = product["current_stock"] - quantity; execute("UPDATE products SET current_stock=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (new_stock, product["id"], g.user["id"])); execute("INSERT INTO inventory_transactions (user_id,product_id,transaction_type,quantity,reference_id,notes) VALUES (?,?,?,?,?,?)", (g.user["id"], product["id"], "SALE", -quantity, cursor.lastrowid, "Recorded sale"))
        sync_reorder_alert({**dict(product), "current_stock": new_stock}, inventory_row({**dict(product), "current_stock": new_stock}))
        return jsonify({"id": cursor.lastrowid, "product_id": product["id"], "quantity": quantity, "current_stock": new_stock}), 201
    except (ValueError, TypeError, KeyError) as exc: return error(str(exc))


@app.post("/api/sales/import")
@login_required
def api_import_sales():
    upload = request.files.get("file")
    if upload is None:
        return error("Select a CSV or XLSX file")
    try:
        filename, frame = read_import_file(upload)
        products = query("SELECT id,name,unit FROM products WHERE user_id=? AND is_active=1", (g.user["id"],))
        existing_keys = {(row["sale_date"], row["product_id"], float(row["quantity"]), str(row["unit"]).casefold(), str(row["selling_price"] or ""), str(row["buyer"] or "").casefold()) for row in query("SELECT sale_date,product_id,quantity,unit,selling_price,buyer FROM sales WHERE user_id=?", (g.user["id"],))}
        rows = validate_sales(frame, products, existing_keys)
        summary = _import_summary(rows)
        summary["filename"] = filename
        if request.form.get("action", "preview") != "confirm":
            return jsonify({"success": True, "preview": True, **summary})
        valid_rows = [row for row in rows if not row["errors"]]
        if not valid_rows:
            return jsonify({"success": True, "imported": 0, "skipped": len(rows), "summary": summary})
        connection = get_db()
        try:
            connection.executemany("INSERT INTO sales (user_id,product_id,sale_date,quantity,unit,selling_price,buyer,notes,entry_method) VALUES (?,?,?,?,?,?,?,?,?)", [(g.user["id"], row["product_id"], row["sale_date"], row["quantity"], row["unit"], row["selling_price"], row["buyer"], "Imported from " + filename, "Historical Import") for row in valid_rows])
            connection.commit()
        except Exception:
            connection.rollback()
            app.logger.exception("Sales import failed")
            return error("Sales could not be imported", 500)
        return jsonify({"success": True, "imported": len(valid_rows), "skipped": len(rows) - len(valid_rows), "summary": summary})
    except ValueError as exc:
        return error(str(exc), 422)


@app.get("/api/sales/suggestion/<int:product_id>")
@login_required
def api_sales_suggestion(product_id):
    product = user_product(product_id)
    suggestion = sales_suggestion(product["id"])
    if suggestion is None:
        return jsonify({"available": False, "message": "Automatic suggestion unavailable. Record more sales history to generate a historical sales suggestion."})
    return jsonify({"available": True, "suggested_quantity": suggestion, "unit": product["unit"], "method": "Weighted average of up to seven recent actual sales"})


@app.route("/inventory/stock-in", methods=("GET", "POST"))
@login_required
def stock_in():
    products = query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))
    if request.method == "POST":
        try:
            product = user_product(int(request.form["product_id"])); quantity = float(request.form["quantity"]); stock_date = request.form.get("stock_date") or date.today().isoformat()
            if quantity <= 0: raise ValueError("Quantity received must be positive")
            stock_date = validate_past_or_today(stock_date, "stock-in")
            new_stock = product["current_stock"] + quantity
            execute("UPDATE products SET current_stock=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (new_stock, product["id"], g.user["id"]))
            execute("INSERT INTO inventory_transactions (user_id,product_id,transaction_type,quantity,notes) VALUES (?,?,?,?,?)", (g.user["id"], product["id"], "STOCK_IN", quantity, request.form.get("notes", "")))
            sync_reorder_alert({**dict(product), "current_stock": new_stock}, inventory_row({**dict(product), "current_stock": new_stock}))
            flash("Stock received and inventory updated.", "success"); return redirect(url_for("inventory_page"))
        except (ValueError, TypeError, KeyError) as exc: flash(str(exc), "error")
    return render_template("stock_in.html", products=products, page_title="Add Stock")


@app.post("/api/inventory/stock-in")
@login_required
def api_stock_in():
    payload = request.get_json(silent=True) or {}
    try:
        product = user_product(int(payload["product_id"])); quantity = float(payload["quantity"]); stock_date = payload.get("stock_date", date.today().isoformat())
        if quantity <= 0: raise ValueError("Quantity received must be positive")
        stock_date = validate_past_or_today(stock_date, "stock-in")
        new_stock = product["current_stock"] + quantity
        execute("UPDATE products SET current_stock=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", (new_stock, product["id"], g.user["id"]))
        cursor = execute("INSERT INTO inventory_transactions (user_id,product_id,transaction_type,quantity,notes) VALUES (?,?,?,?,?)", (g.user["id"], product["id"], "STOCK_IN", quantity, payload.get("notes", "")))
        sync_reorder_alert({**dict(product), "current_stock": new_stock}, inventory_row({**dict(product), "current_stock": new_stock}))
        return jsonify({"id": cursor.lastrowid, "product_id": product["id"], "current_stock": new_stock}), 201
    except (ValueError, TypeError, KeyError) as exc: return error(str(exc))


@app.get("/api/inventory")
@login_required
def api_inventory(): return jsonify([inventory_row(row) for row in query("SELECT * FROM products WHERE user_id=? AND is_active=1 ORDER BY name", (g.user["id"],))])


@app.get("/api/inventory/<int:product_id>")
@login_required
def api_inventory_product(product_id):
    try:
        product = user_product(product_id)
        result = inventory_row(product)
        result["transactions"] = [dict(row) for row in query("SELECT * FROM inventory_transactions WHERE user_id=? AND product_id=? ORDER BY created_at DESC,id DESC", (g.user["id"], product_id))]
        return jsonify(result)
    except ValueError as exc: return error(str(exc), 404)


@app.get("/api/alerts")
@login_required
def api_alerts():
    sync_user_reorder_alerts()
    return jsonify([dict(row) for row in query("SELECT a.*,p.name product_name FROM alerts a JOIN products p ON p.id=a.product_id WHERE a.user_id=? ORDER BY a.status ASC, a.created_at DESC", (g.user["id"],))])


@app.put("/api/alerts/<int:alert_id>/read")
@login_required
def api_alert_read(alert_id):
    cursor = execute("UPDATE alerts SET is_read=1 WHERE id=? AND user_id=?", (alert_id, g.user["id"]))
    if cursor.rowcount == 0: return error("Alert not found", 404)
    return jsonify({"id": alert_id, "is_read": True})


@app.get("/api/dashboard-summary")
@login_required
def dashboard_summary():
    return jsonify({"total_products": query("SELECT COUNT(*) total FROM products WHERE user_id=? AND is_active=1", (g.user["id"],), one=True)["total"], "today_sales": query("SELECT COALESCE(SUM(quantity),0) total FROM sales WHERE user_id=? AND sale_date=date('now')", (g.user["id"],), one=True)["total"], "current_stock": query("SELECT COALESCE(SUM(current_stock),0) total FROM products WHERE user_id=? AND is_active=1", (g.user["id"],), one=True)["total"], "active_alerts": query("SELECT COUNT(*) total FROM alerts WHERE user_id=? AND status='ACTIVE'", (g.user["id"],), one=True)["total"]})


@app.get("/api/forecasts/<int:product_id>")
@login_required
def api_user_forecast(product_id):
    product = user_product(product_id)
    sales = query("SELECT sale_date,quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date<=date('now') ORDER BY sale_date", (g.user["id"], product_id))
    frame = pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity"))
    status = customer_model_status(frame, g.user["id"], product_id)
    return jsonify({"success": True, "product_id": product["id"], **status})


@app.post("/api/forecasts/<int:product_id>/train")
@login_required
def api_train_customer_forecast(product_id):
    product = user_product(product_id)
    sales = query("SELECT sale_date,quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date<=date('now') ORDER BY sale_date", (g.user["id"], product_id))
    frame = pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity"))
    try:
        metadata = train_customer_lstm(frame, g.user["id"], product_id)
        return jsonify({"status": "trained", "product_id": product["id"], "metadata": metadata}), 201
    except ValueError as exc:
        return error(str(exc), 422)


@app.post("/api/forecasts/<int:product_id>/generate")
@login_required
def api_generate_customer_forecast(product_id):
    product = user_product(product_id)
    payload = request.get_json(silent=True) or {}
    try:
        horizon = int(payload.get("horizon", 7))
        if horizon not in (7, 14, 30, 90, 180): raise ValueError("Forecast horizon must be 7, 14, 30, 90, or 180 days")
        sales = query("SELECT sale_date,quantity FROM sales WHERE user_id=? AND product_id=? AND sale_date<=date('now') ORDER BY sale_date", (g.user["id"], product_id))
        frame = pd.DataFrame([dict(row) for row in sales], columns=("sale_date", "quantity"))
        if frame.empty:
            raise ValueError("Insufficient historical data to generate a customer-specific forecast.")
        forecast = forecast_customer_lstm(frame, g.user["id"], product_id, horizon)
        for item in forecast:
            execute("INSERT INTO forecasts (user_id,product_id,forecast_date,forecast_value,model_name) VALUES (?,?,?,?,?)", (g.user["id"], product_id, item["Date"], item["Predicted Demand"], "Customer LSTM"))
        return jsonify({"success": True, "product_id": product["id"], "product": product["name"], "model_used": "Customer LSTM", "forecast": forecast})
    except ValueError as exc:
        return error(str(exc), 422)
    except Exception:
        app.logger.exception("Customer forecast generation failed")
        return error("Forecast could not be generated", 500)


# Existing dataset-backed APIs remain available for the original demonstration workflow.
@app.get("/legacy")
def legacy_index(): return render_template("index.html")


@app.get("/api/stores")
def stores(): return jsonify(sorted(DATA["Store ID"].unique().tolist()))


@app.get("/api/legacy-products")
def legacy_products(): return jsonify(sorted(DATA["Product ID"].unique().tolist()))


@app.get("/api/model-metrics")
def model_metrics(): return jsonify({"best_model": BEST_MODEL, "metrics": METRICS.to_dict(orient="records")})


def parse_request():
    payload = request.get_json(silent=True) or {}; store_id, product_id = payload.get("store_id"), payload.get("product_id")
    try: horizon = int(payload.get("horizon", DEFAULT_FORECAST_HORIZON))
    except (TypeError, ValueError): raise ValueError("horizon must be an integer")
    if store_id not in set(DATA["Store ID"]): raise ValueError("Unknown store")
    if product_id not in set(DATA["Product ID"]): raise ValueError("Unknown product")
    if horizon not in (7, 14, 30): raise ValueError("horizon must be 7, 14, or 30")
    return payload, store_id, product_id, horizon


def legacy_forecast_payload():
    payload, store_id, product_id, horizon = parse_request(); model_name = payload.get("model") or BEST_MODEL
    if model_name == "LSTM" and LSTM_MODEL is not None: predictions = recursive_lstm_forecast(DATA, store_id, product_id, horizon, LSTM_MODEL, LSTM_SCALER["scaler"], LSTM_SCALER["lookback"])
    else:
        if model_name not in MODELS: model_name = "Random Forest"
        predictions = recursive_tree_forecast(DATA, store_id, product_id, horizon, MODELS[model_name])
    history = DATA[(DATA["Store ID"] == store_id) & (DATA["Product ID"] == product_id)].tail(30)
    return {"store_id": store_id, "product_id": product_id, "model_used": model_name, "historical": [{"Date": row.Date.strftime("%Y-%m-%d"), "Demand": float(row.Demand)} for row in history.itertuples()], "forecast": predictions}


@app.post("/api/forecast")
def forecast():
    try:
        payload = legacy_forecast_payload(); payload["success"] = True
        return jsonify(payload)
    except ValueError as exc: return error(str(exc), 400)
    except Exception: app.logger.exception("Forecast failed"); return error("Forecast could not be generated", 500)


@app.post("/api/inventory-recommendation")
def legacy_inventory():
    try:
        payload = legacy_forecast_payload(); series = DATA[(DATA["Store ID"] == payload["store_id"]) & (DATA["Product ID"] == payload["product_id"])]
        options = request.get_json(silent=True) or {}; result = inventory_recommendation(sum(item["Predicted Demand"] for item in payload["forecast"]), float(series["Inventory Level"].iloc[-1]), float(series["Demand"].tail(30).std()), int(options.get("lead_time", DEFAULT_LEAD_TIME)), float(options.get("service_level", DEFAULT_SERVICE_LEVEL)))
        return jsonify({**payload, **result})
    except (ValueError, TypeError) as exc: return error(str(exc), 400)
    except Exception: app.logger.exception("Inventory calculation failed"); return error("Inventory recommendation could not be generated", 500)


if __name__ == "__main__": app.run(debug=True)
