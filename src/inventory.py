import math
from statistics import NormalDist


def inventory_recommendation(forecast_demand, current_inventory, demand_std, lead_time=7, service_level=0.95, review_period=None):
    if lead_time <= 0 or not 0 < service_level < 1:
        raise ValueError("lead_time must be positive and service_level must be between 0 and 1")
    review_period = lead_time if review_period is None else review_period
    z = NormalDist().inv_cdf(service_level)
    safety_stock = z * max(float(demand_std), 0.0) * math.sqrt(lead_time)
    average_daily = float(forecast_demand) / max(int(review_period), 1)
    reorder_point = average_daily * lead_time + safety_stock
    target_stock = float(forecast_demand) + safety_stock
    reorder_quantity = max(0.0, target_stock - float(current_inventory))
    return {"current_inventory": float(current_inventory), "forecast_demand": float(forecast_demand), "lead_time": int(lead_time), "service_level": float(service_level), "safety_stock": float(safety_stock), "reorder_point": float(reorder_point), "suggested_reorder_quantity": float(reorder_quantity), "recommendation": "REORDER" if current_inventory <= reorder_point else "NO REORDER"}