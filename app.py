
import re
import streamlit as st
from datetime import date, timedelta
from smartroad_core import predict_traffic, predict_risk

st.set_page_config(page_title="SmartRoad AI", page_icon="🚦")
st.title("🚦 SmartRoad AI")
st.caption("Predict. Prepare. Travel Safer.")

tab1, tab2, tab3 = st.tabs(["🚦 Traffic Intelligence", "🛵 Two-Wheeler Safety", "🤖 AI Assistant"])

with tab1:
    d = st.date_input("Select date", date.today())
    if st.button("Predict traffic"):
        t = predict_traffic(d.isoformat())
        st.metric("Peak traffic time", t["peak"])
        st.metric("Longest heavy period (hours)", t["duration"])
        st.write("**Heavy traffic periods:**", ", ".join(t["heavy"]))
        st.success("Recommended travel window: " + t["best"])

with tab2:
    d2 = st.date_input("Travel date", date.today(), key="d2")
    hour = st.slider("Travel hour (0-23)", 0, 23, 22)
    weather = st.selectbox("Weather", ["clear", "rain", "fog"])
    road = st.selectbox("Road type", ["city", "highway"])
    junction = st.checkbox("Junction on the route")
    lit = st.checkbox("Street is well lit", value=True)
    if st.button("Check safety risk"):
        r = predict_risk(d2.isoformat(), hour, weather, road,
                         junction=int(junction), street_lit=int(lit))
        if r["risk"] == "High":
            st.error("Two-Wheeler Safety Risk: HIGH")
        elif r["risk"] == "Moderate":
            st.warning("Two-Wheeler Safety Risk: MODERATE")
        else:
            st.success("Two-Wheeler Safety Risk: LOW")
        st.write("**Traffic at that time:**", r["traffic"])
        st.write("**Confidence:**", r["confidence"])
        st.write("**Contributing factors:**")
        for f in r["factors"]:
            st.write("- " + f)
        st.info(r["advice"])

def understand(q):
    s = q.lower()
    d = date.today() + timedelta(days=1 if "tomorrow" in s else 0)
    m = re.search(r"(\d{1,2})\s*(am|pm)", s)
    hour = 12
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
    weather = "rain" if "rain" in s else "fog" if "fog" in s else "clear"
    road = "highway" if "highway" in s else "city"
    junction = 1 if "junction" in s else 0
    lit = 0 if ("dark" in s or "unlit" in s) else 1
    return dict(date=d.isoformat(), hour=hour, weather=weather,
                road_type=road, junction=junction, street_lit=lit)

with tab3:
    q = st.text_input("Ask SmartRoad AI", "I need to travel by bike at 11 PM tomorrow on the highway. It's raining. Is it a good time?")
    if st.button("Ask"):
        with st.status("Agent working...", expanded=True) as status:
            st.write("🧠 Understanding request...")
            p = understand(q)
            st.json(p)
            st.write("🔧 Calling traffic tool...")
            t = predict_traffic(p["date"])
            st.write("🔧 Calling safety tool...")
            r = predict_risk(p["date"], p["hour"], p["weather"], p["road_type"],
                             junction=p["junction"], street_lit=p["street_lit"])
            status.update(label="Done", state="complete")
        st.subheader("💡 Recommendation")
        st.write(f"Predicted two-wheeler risk: **{r['risk']}** (confidence {r['confidence']}). "
                 f"Main factors: {', '.join(r['factors']) or 'none significant'}. "
                 f"Best travel window: {t['best']}. {r['advice']}")
