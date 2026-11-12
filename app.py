import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visitor Purchase Intent", page_icon="🛒")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


bundle = load_model()
model, thr = bundle["pipeline"], bundle["threshold"]

st.title("🛒 Will this visitor buy?")
st.caption("Enter a visitor's browsing session to estimate their chance of buying.")

c1, c2 = st.columns(2)
with c1:
    product_pages = st.number_input("Product pages viewed", 0, 700, 10)
    product_time = st.number_input("Time on product pages (seconds)", 0.0, 64000.0, 300.0)
    page_values = st.number_input("Page value", 0.0, 400.0, 0.0)
with c2:
    bounce = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.001)
    exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, 0.001)
    month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul",
                                   "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)

visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
weekend = st.checkbox("Weekend visit")

with st.expander("Advanced details (optional)"):
    admin_pages = st.number_input("Account pages viewed", 0, 30, 0)
    admin_time = st.number_input("Time on account pages (s)", 0.0, 3500.0, 0.0)
    info_pages = st.number_input("Info pages viewed", 0, 25, 0)
    info_time = st.number_input("Time on info pages (s)", 0.0, 2600.0, 0.0)
    special_day = st.slider("Closeness to a special day", 0.0, 1.0, 0.0, 0.2)
    os_ = st.number_input("Operating system code", 1, 8, 2)
    browser = st.number_input("Browser code", 1, 13, 2)
    region = st.number_input("Region code", 1, 9, 1)
    traffic = st.number_input("Traffic type code", 1, 20, 2)

if st.button("Predict", type="primary"):
    try:
        row = pd.DataFrame([{
            "Administrative": admin_pages, "Administrative_Duration": admin_time,
            "Informational": info_pages, "Informational_Duration": info_time,
            "ProductRelated": product_pages, "ProductRelated_Duration": product_time,
            "BounceRates": bounce, "ExitRates": exit_rate, "PageValues": page_values,
            "SpecialDay": special_day, "Month": month,
            "OperatingSystems": int(os_), "Browser": int(browser),
            "Region": int(region), "TrafficType": int(traffic),
            "VisitorType": visitor, "Weekend": bool(weekend),
        }])
        p = model.predict_proba(row)[0, 1]
        st.metric("Purchase probability", f"{p:.0%}")

        if p >= thr:
            st.success("Likely to buy → no action needed (don't waste a discount).")
        elif p >= thr / 2:
            st.warning("Undecided → show a discount coupon.")
        else:
            st.error("Unlikely to buy → offer the chat assistant.")
    except Exception as e:
        st.error(f"Something went wrong, please check your inputs. ({e})")
