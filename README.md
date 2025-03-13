
## Doctor Activity Predictor

##  Overview
The **Doctor Activity Predictor** is a **machine learning-powered web application** that predicts which doctors are most likely to take a survey at a given time. Users input a time (e.g., `"06:00"`), and the app generates an **Excel file** containing a list of NPIs (doctor IDs) who are predicted to be active.

##  Features
 **Machine Learning Model**: Uses a trained `RandomForestClassifier` to predict active doctors.  
 **User-Friendly Web Interface**: Built with **Streamlit** for easy interaction.  
 **Excel File Export**: Downloads the results in `.xlsx` format.  
 **Real-Time Predictions**: Users can input any time and get instant results.  
  

---
##  Project Structure
```
Doctor-Activity-Predictor/
│── model.pkl                 # Trained Machine Learning model
│── dummy_npi_data.xlsx        # Dataset (Excel format)
│── main.py                    # Streamlit web app
│── doctors.py                 # Model training script (from Colab)
│── README.md                  # Documentation (this file)
│── requirements.txt            # List of dependencies
