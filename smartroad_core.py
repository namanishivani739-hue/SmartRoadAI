import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

T = joblib.load(os.path.join(BASE_DIR, "traffic_model.pkl"))
R = joblib.load(os.path.join(BASE_DIR, "risk_model.pkl"))

W_TRAFFIC = {"clear": "Clear", "cloudy": "Clouds", "rain": "Rain", "fog": "Fog"}
W_CLOUDS = {"clear": 10, "cloudy": 75, "rain": 90, "fog": 90}

def fmt(h):
    h %= 24
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"

def level_of(v):
    return "Low" if v < T["low"] else ("Medium" if v < T["high"] else "Heavy")

def runs(hourly, wanted, lo=0, hi=24):
    out, start = [], None
    for h in range(lo, hi):
        ok = hourly[h]["level"] in wanted
        if ok and start is None:
            start = h
        if (not ok) and start is not None:
            out.append((start, h)); start = None
    if start is not None:
        out.append((start, hi))
    return out

def predict_traffic(date, weather="clear"):
    d = pd.to_datetime(date)
    rows = []
    for h in range(24):
        r = dict.fromkeys(T["columns"], 0)
        r.update(hour=h, dayofweek=d.dayofweek, month=d.month,
                 is_weekend=int(d.dayofweek >= 5), is_holiday=0, temp=300,
                 rain_1h=2.0 if weather == "rain" else 0.0, clouds_all=W_CLOUDS[weather])
        col = "w_" + W_TRAFFIC[weather]
        if col in r:
            r[col] = 1
        rows.append(r)
    vol = T["model"].predict(pd.DataFrame(rows)[T["columns"]])
    hourly = [{"hour": h, "volume": int(v), "level": level_of(v)} for h, v in enumerate(vol)]
    heavy = runs(hourly, {"Heavy"})
    best = runs(hourly, {"Low"}, 6, 22) or runs(hourly, {"Medium"}, 6, 22)
    best = max(best, key=lambda x: x[1] - x[0]) if best else None
    return {
        "date": str(d.date()), "weather": weather, "hourly": hourly,
        "peak": fmt(max(hourly, key=lambda x: x["volume"])["hour"]),
        "heavy": [f"{fmt(a)} to {fmt(b)}" for a, b in heavy],
        "duration": max([b - a for a, b in heavy], default=0),
        "best": f"{fmt(best[0])} to {fmt(best[1])}" if best else "No clear window",
    }

def predict_risk(date, hour, weather="clear", road_type="arterial", junction=0, street_lit=1, traffic=None):
    d = pd.to_datetime(date)
    if traffic is None:
        traffic = predict_traffic(date, weather)["hourly"][hour]["level"]
    row = dict.fromkeys(R["columns"], 0)
    row.update(hour=hour, dayofweek=d.dayofweek, junction=junction, street_lit=street_lit)
    for k in ("weather_" + weather, "traffic_" + traffic, "road_type_" + road_type):
        if k in row:
            row[k] = 1
    X = pd.DataFrame([row])[R["columns"]]
    proba = R["model"].predict_proba(X)[0]
    classes = list(R["model"].classes_)
    label = classes[proba.argmax()]
    night = hour >= 22 or hour < 5
    factors = []
    if night: factors.append("night-time riding")
    if weather in ("rain", "fog"): factors.append(f"{weather} reduces grip and visibility")
    if road_type == "highway": factors.append("high-speed highway")
    if junction: factors.append("junction on the route")
    if night and not street_lit: factors.append("poorly lit road")
    if traffic == "Heavy": factors.append("heavy traffic")
    advice = {
        "High": "Avoid this time if you can. If you must ride, use a lit main road, slow down and wear full safety gear.",
        "Moderate": "Ride with extra care and check the weather and lighting before you leave.",
        "Low": "Conditions look fine. Follow normal safety rules.",
    }[label]
    return {"risk": label, "traffic": traffic, "confidence": round(float(proba.max()), 2),
            "factors": factors, "advice": advice}
