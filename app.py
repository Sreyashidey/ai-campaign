import streamlit as st
import pandas as pd
from doctor import DoctorActivityModel
from datetime import datetime
import numpy as np
import altair as alt

# Initialize the model
model_handler = DoctorActivityModel()
model_handler.load_model("doctor_activity_model.pkl")

# Load Excel dataset
@st.cache_data
def load_data():
    return pd.read_excel("dummy_npi_data.xlsx")

df = load_data()

# ---- Custom CSS for Dark UI ----
st.markdown("""
    <style>
    /* Black background with modern UI */
    .stApp {
        background: #000000;
        color: #EAEAEA;  /* Light text for contrast */
        font-family: 'Poppins', sans-serif;
    }
    /* Stylish heading */
    .main-title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #F5F5F5;  /* Soft white shade */
        text-shadow: 2px 2px 6px rgba(255, 255, 255, 0.2);
    }
    /* Custom text styling */
    .custom-text {
        font-size: 18px;
        color: #D3D3D3;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Custom input field */
    .css-1d391kg {
        border-radius: 10px;
        padding: 8px;
        font-size: 16px;
        border: 2px solid #F5F5F5;
        background-color: #222;
        color: #FFF;
    }
    /* Custom button */
    .stButton>button {
        background-color: #FF5733;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #E64A19;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- Page Title & Subtitle ----
st.markdown("<h1 class='main-title'>Doctor Activity Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='custom-text'>Enter a time to predict active doctors and explore insightful visualizations.</p>", unsafe_allow_html=True)

# ---- User Input ----
input_time = st.text_input("Enter Time", "06:00")

# ---- Validate Input Time Format ----
def validate_time(input_time):
    try:
        datetime.strptime(input_time, "%H:%M")
        return True
    except ValueError:
        return False

# ---- Doctor Activity Chart ----
st.markdown("Doctor Activity Over 24 Hours")
chart_data = pd.DataFrame({
    'Time': [f"{hour:02d}:00" for hour in range(0, 24)],
    'Activity': np.random.randint(20, 100, 24)
})

activity_chart = alt.Chart(chart_data).mark_line(point=True, color="#FF5733").encode(
    x=alt.X('Time', title='Time of Day', axis=alt.Axis(labelColor="white", titleColor="white")),
    y=alt.Y('Activity', title='Activity Level', axis=alt.Axis(labelColor="white", titleColor="white")),
    tooltip=['Time', 'Activity']
).properties(
    width=700,
    height=400,
    background="#000000"
).configure_axis(
    gridColor="gray"
)

st.altair_chart(activity_chart, use_container_width=True)

# ---- Predict & Export Button ----
if st.button("Predict & Export"):
    if not validate_time(input_time):
        st.error("Invalid time format. Please enter time in HH:MM format.")
    else:
        hour = int(input_time.split(":")[0])
        recommended_doctors = model_handler.predict_active_doctors(df, hour)

        if recommended_doctors.empty:
            st.warning("No active doctors found for the given time.")
        else:
            st.markdown("Recommended Doctors by Specialty")
            
            # If 'specialty' column exists, display a bar chart
            if "specialty" in recommended_doctors.columns:
                spec_data = recommended_doctors['specialty'].value_counts().reset_index()
                spec_data.columns = ['Specialty', 'Count']
                specialty_chart = alt.Chart(spec_data).mark_bar(color="#FF5733").encode(
                    x=alt.X('Specialty:N', sort='-y', title='Specialty', axis=alt.Axis(labelColor="white", titleColor="white")),
                    y=alt.Y('Count:Q', title='Number of Doctors', axis=alt.Axis(labelColor="white", titleColor="white")),
                    tooltip=['Specialty', 'Count']
                ).properties(
                    width=700,
                    height=400,
                    background="#000000"
                ).configure_axis(
                    gridColor="gray"
                )

                st.altair_chart(specialty_chart, use_container_width=True)
            else:
                st.info("No specialty information available for visualization.")

            # Save as Excel
            output_file = "recommended_doctors.xlsx"
            recommended_doctors.to_excel(output_file, index=False)

            # Download Button
            st.success("Prediction Complete! Download the Excel file below.")
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Excel",
                    data=file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )