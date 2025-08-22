import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==============================
# Dataset
# ==============================
original_data = pd.DataFrame({
    'distance': [0.1, 1.5, 2.8, 3.9, 6.5, 8.2, 10.7, 11.5, 12.8],
    'rssi':     [-83.4, -90, -96, -103.2, -100, -102, -104.1, -112, -126.6]
})

try:
    user_data = pd.read_csv("rssi_dataset.csv")
    combined_data = pd.concat([original_data, user_data], ignore_index=True)
except Exception:
    combined_data = original_data

X = combined_data[['distance']]
y = combined_data['rssi']

# ==============================
# Train Models
# ==============================
rssi_model = RandomForestRegressor(n_estimators=200, random_state=42)
rssi_model.fit(X, y)

boost_data = pd.DataFrame({
    'RSSI': [-83.4, -90, -96, -100, -104, -112, -126],
    'Boost_dBm': [0, 3, 5, 7, 9, 12, 15]
})
boost_model = RandomForestRegressor(n_estimators=200, random_state=42)
boost_model.fit(boost_data[['RSSI']], boost_data['Boost_dBm'])

rssi_r2 = r2_score(y, rssi_model.predict(X))
boost_r2 = r2_score(boost_model.predict(boost_data[['RSSI']]), boost_data['Boost_dBm'])

# ==============================
# UI
# ==============================
st.title("5G Signal Optimizer (Extended Data Model)")

distance = st.slider("Distance from gNB (meters)", 0.1, 13.0, 5.0, step=0.1)
obstacles = {
    "None (LOS)": 0,
    "Single table": -3,
    "Double table": -4,
    "Painted brick wall": -6,
    "Indoor wall": -5,
}
obstacle = st.selectbox("Obstacle Type", list(obstacles.keys()))
obstacle_loss = obstacles[obstacle]

# ==============================
# Prediction
# ==============================
predicted_rssi = rssi_model.predict(np.array([[distance]]))[0] + obstacle_loss
predicted_boost = boost_model.predict(np.array([[predicted_rssi]]))[0]
improved_rssi = predicted_rssi + predicted_boost

def classify_signal(rssi):
    if rssi > -90:
        return "Good"
    elif rssi > -105:
        return "Fair"
    else:
        return "Poor"

signal_quality_before = classify_signal(predicted_rssi)
signal_quality_after = classify_signal(improved_rssi)

# ==============================
# Results
# ==============================
st.markdown("### Prediction Results")
st.write(f"- **Predicted RSSI (with obstacle):** {predicted_rssi:.2f} dBm")
st.write(f"- **Recommended Boost:** +{predicted_boost:.2f} dBm")
st.write(f"- **Improved RSSI after Boost:** {improved_rssi:.2f} dBm")
st.write(f"- **Signal Quality before Boost:** {signal_quality_before}")
st.write(f"- **Signal Quality after Boost:** {signal_quality_after}")

# ==============================
# Plot Horizontal Levels
# ==============================
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(y=predicted_rssi, color='blue', linewidth=2,
           label=f"Predicted RSSI: {predicted_rssi:.2f} dBm")
ax.axhline(y=improved_rssi, color='green', linestyle='--', linewidth=2,
           label=f"Boosted RSSI: {improved_rssi:.2f} dBm")
#ax.axhline(y=-90, color='orange', linestyle='dashed', linewidth=2,label="Threshold: -90 dBm")
ax.set_ylim(-140, -70)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_ylabel("RSSI (dBm)")
ax.set_title("Signal Strength Levels")
ax.grid(True)
ax.legend(loc='upper right')
st.pyplot(fig)

# ==============================
# Curve Plot
# ==============================
if st.checkbox("Show RSSI vs Distance Curve"):
    st.markdown("#### Model Prediction Across Distance")
    test_distances = np.linspace(0.1, 13, 300).reshape(-1, 1)
    predicted_rssi_curve = rssi_model.predict(test_distances)
    boost_curve = boost_model.predict(predicted_rssi_curve.reshape(-1, 1))
    improved_rssi_curve = predicted_rssi_curve + boost_curve

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(test_distances, predicted_rssi_curve,
             label="Predicted RSSI", color="blue")
    ax2.plot(test_distances, improved_rssi_curve,
             label="Boosted RSSI", color="green", linestyle="--")
    ax2.scatter([distance], [predicted_rssi], color='red', label="Your RSSI")
    ax2.scatter([distance], [improved_rssi], color='darkgreen',
                label="Your Boosted RSSI")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("RSSI (dBm)")
    ax2.set_title("Model Prediction Across Distance")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ==============================
# Accuracy
# ==============================
st.markdown("### Model Accuracy")
st.write(f"- **RSSI Model R² Score:** `{rssi_r2:.2f}`")
st.write(f"- **Boost Model R² Score:** `{boost_r2:.2f}`")

# ==============================
# Export to CSV
# ==============================
if st.button("Download Prediction as CSV"):
    result_df = pd.DataFrame({
        "Distance (m)": [distance],
        "Obstacle": [obstacle],
        "Predicted RSSI (dBm)": [predicted_rssi],
        "Recommended Boost (dBm)": [predicted_boost],
        "Improved RSSI (dBm)": [improved_rssi],
        "Signal Quality Before": [signal_quality_before],
        "Signal Quality After": [signal_quality_after]
    })
    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    st.download_button("Download", buffer.getvalue(),
                       file_name="5g_signal_prediction.csv", mime="text/csv")

st.caption("Upload 'rssi_dataset.csv' to extend training data and improve predictions.")
