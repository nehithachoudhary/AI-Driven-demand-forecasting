from datetime import date, datetime
from io import BytesIO
import re

import pandas as pd


ALLOWED_EXTENSIONS = {"csv", "xlsx"}
PRODUCT_ALIASES = {
    "name": {"productname", "product", "itemname", "item", "name"},
    "category": {"category", "productcategory", "type"},
    "unit": {"unit", "measurement", "uom"},
    "current_stock": {"currentstock", "stock", "quantity", "inventory", "availablestock"},
    "region": {"region", "area"},
    "safety_stock": {"safetystock", "minimumstock"},
    "lead_time_days": {"leadtimedays", "leadtime"},
}
SALES_ALIASES = {
    "sale_date": {"date", "saledate", "salesdate", "transactiondate"},
    "product": {"product", "productname", "item", "itemname"},
    "quantity": {"quantity", "quantitysold", "unitssold", "soldquantity", "qty"},
    "unit": {"unit", "measurement", "uom"},
    "selling_price": {"amount", "salesamount", "totalamount"},
    "buyer": {"buyer", "customer", "customername"},
}


def _clean_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _column_map(frame, aliases, required):
    columns = {}
    for column in frame.columns:
        normalized = _clean_header(column)
        matches = [name for name, options in aliases.items() if normalized in options]
        if len(matches) == 1:
            if matches[0] in columns:
                raise ValueError(f"Multiple columns map to {matches[0]}")
            columns[matches[0]] = column
    missing = [field for field in required if field not in columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return columns


def read_import_file(file_storage):
    filename = (file_storage.filename or "").strip()
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only CSV and XLSX files are supported")
    content = file_storage.read()
    if not content:
        raise ValueError("The uploaded file is empty")
    try:
        if extension == "csv":
            frame = pd.read_csv(BytesIO(content))
        else:
            frame = pd.read_excel(BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise ValueError("The uploaded file could not be read") from exc
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The uploaded file contains no rows")
    return filename, frame.dropna(how="all").reset_index(drop=True)


def _number(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric")
    if not pd.notna(number):
        raise ValueError(f"{field} must be numeric")
    return number


def validate_products(frame, user_region, existing_names):
    mapping = _column_map(frame, PRODUCT_ALIASES, ("name", "unit", "current_stock"))
    seen = set()
    rows = []
    for index, source in frame.iterrows():
        row_number = index + 2
        values = {field: _clean_text(source[column]) for field, column in mapping.items()}
        errors = []
        name_key = values["name"].casefold()
        if not values["name"] or len(values["name"]) < 2:
            errors.append("Product name is required")
        if not values["unit"]:
            errors.append("Unit is required")
        try:
            stock = _number(values["current_stock"], "Current stock")
            if stock < 0:
                errors.append("Current stock cannot be negative")
        except ValueError as exc:
            errors.append(str(exc))
            stock = 0
        if name_key in seen or name_key in existing_names:
            errors.append("Product already exists")
        seen.add(name_key)
        row = {"row": row_number, "name": values["name"], "category": values.get("category") or "General", "unit": values["unit"], "region": values.get("region") or user_region, "current_stock": stock, "safety_stock": 0, "lead_time_days": 7, "errors": errors}
        rows.append(row)
    return rows


def _parse_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Invalid sale date")
    parsed_date = parsed.date()
    if parsed_date > date.today():
        raise ValueError("Sales date cannot be in the future")
    return parsed_date.isoformat()


def validate_sales(frame, products, existing_keys):
    mapping = _column_map(frame, SALES_ALIASES, ("sale_date", "product", "quantity"))
    product_map = {str(row["name"]).strip().casefold(): row for row in products}
    seen = set()
    rows = []
    for index, source in frame.iterrows():
        row_number = index + 2
        values = {field: _clean_text(source[column]) for field, column in mapping.items()}
        errors = []
        try:
            sale_date = _parse_date(values["sale_date"])
        except ValueError as exc:
            sale_date = ""
            errors.append(str(exc))
        product = product_map.get(values["product"].casefold())
        if product is None:
            errors.append(f"Product '{values['product']}' is not registered")
        try:
            quantity = _number(values["quantity"], "Quantity")
            if quantity <= 0:
                errors.append("Quantity must be greater than zero")
        except ValueError as exc:
            quantity = 0
            errors.append(str(exc))
        if product and values.get("unit") and values["unit"].casefold() != str(product["unit"]).casefold():
            errors.append(f"Unit must match product unit ({product['unit']})")
        key = (sale_date, product["id"] if product else values["product"].casefold(), quantity, values.get("unit", "").casefold(), values.get("selling_price", ""), values.get("buyer", "").casefold())
        if key in seen or key in existing_keys:
            errors.append("Duplicate sale record")
        seen.add(key)
        rows.append({"row": row_number, "sale_date": sale_date, "product_id": product["id"] if product else None, "product": values["product"], "quantity": quantity, "unit": values.get("unit") or (product["unit"] if product else ""), "selling_price": values.get("selling_price") or None, "buyer": values.get("buyer", ""), "errors": errors})
    return rows
