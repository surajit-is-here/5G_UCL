# signal_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Prepare ML models
# ----------------------------

# RSSI dataset (distance vs signal)
distances = np.array([0.1, 1.5, 2.8, 3.9, 6.5, 8.2, 10.7, 11.5, 12.8]).reshape(-1, 1)
rssi_values = np.array([-83.4, -90, -96, -103.2, -100, -102, -104.1, -112, -126.6])

# Train RSSI model
rssi_model = RandomForestRegressor(n_estimators=200, random_state=42)
rssi_model.fit(distances, rssi_values)

# Boost dataset (RSSI vs required dBm gain)
boost_training = pd.DataFrame({
    'RSSI': [-83.4, -90, -96, -100, -104, -112, -126],
    'Boost_dBm': [0, 3, 5, 7, 9, 12, 15]
})

boost_model = RandomForestRegressor(n_estimators=200, random_state=42)
boost_model.fit(boost_training[['RSSI']], boost_training['Boost_dBm'])

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("📶 5G Signal Optimizer Dashboard (Simulated)")
st.markdown("This dashboard predicts RSSI and required boost based on distance and obstacles.")

# User inputs
distance = st.slider("📏 Distance from gNB (in meters)", 0.1, 15.0, 5.0, step=0.1)

obstacles = {
    "None (LOS)": 0,
    "Single table": -3,
    "Double table": -4,
    "Painted brick wall": -6,
    "Indoor wall": -5,
}

obstacle = st.selectbox("🧱 Obstacle Type", list(obstacles.keys()))
obstacle_loss = obstacles[obstacle]

# Prediction
predicted_rssi = rssi_model.predict(np.array([[distance]]))[0] + obstacle_loss
predicted_boost = boost_model.predict(np.array([[predicted_rssi]]))[0]
improved_rssi = predicted_rssi + predicted_boost

# Output
st.markdown("### 🔍 Prediction Results:")
st.write(f"- **Predicted RSSI:** {predicted_rssi:.2f} dBm")
st.write(f"- **Recommended Boost Power:** +{predicted_boost:.2f} dBm")
st.write(f"- **Improved RSSI after Boost:** {improved_rssi:.2f} dBm")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["Predicted", "After Boost"], [predicted_rssi, improved_rssi], color=["blue", "green"])
ax.axhline(y=-90, color="orange", linestyle="--", label="RSSI = -90 dBm (Threshold)")
ax.set_ylabel("RSSI (dBm)")
ax.set_title("Signal Strength Before and After Boost")
ax.legend()
st.pyplot(fig)

# Footer
st.caption("Note: All calculations are based on synthetic simulation models.")
