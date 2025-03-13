import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class DoctorActivityModel:
    def __init__(self):
        # Initialize variables
        self.model = None
        self.label_encoders = {}

    def preprocess_data(self, df):
        """
        Preprocess the dataset by handling missing values, encoding categorical variables,
        and extracting features.
        """
        # Convert Login Time and Logout Time to datetime
        df['Login Time'] = pd.to_datetime(df['Login Time'], format='%H:%M:%S')
        df['Logout Time'] = pd.to_datetime(df['Logout Time'], format='%H:%M:%S')

        # Extract hour from Login Time
        df['Hour'] = df['Login Time'].dt.hour

        # Fill missing values
        df['Usage Time (mins)'] = df['Usage Time (mins)'].fillna(df['Usage Time (mins)'].median())
        df['Count of Survey Attempts'] = df['Count of Survey Attempts'].fillna(0)

        # Encode categorical variables
        categorical_columns = ['State', 'Region', 'Speciality']
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le  # Save the encoder for later use

        # Recreate Session Duration if needed
        if 'Session Duration' not in df.columns:
            df['Session Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60

        return df

    def train_model(self, df):
        """
        Train a machine learning model to predict whether a doctor is likely to take a survey.
        """
        # Define features and target variable
        features = ['Hour', 'Usage Time (mins)', 'State', 'Region', 'Speciality', 'Count of Survey Attempts']
        target = 'Survey Taken'

        # Create target variable based on Count of Survey Attempts
        df['Survey Taken'] = df['Count of Survey Attempts'].apply(lambda x: 1 if x > 0 else 0)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

    def save_model(self, filepath="doctor_activity_model.pkl"):
        """
        Save the trained model and label encoders to a file.
        """
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="doctor_activity_model.pkl"):
        """
        Load the trained model and label encoders from a file.
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        print(f"Model loaded from {filepath}")

    def predict_active_doctors(self, df, input_hour):
        """
        Predict doctors who are likely to take surveys at a given hour.
        """
        # Ensure the model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded. Please load or train the model first.")

        # Preprocess the data
        df = self.preprocess_data(df)

        # Filter data for the given hour
        df = df[df['Hour'] == input_hour]

        # Predict active doctors
        features_for_prediction = ['Hour', 'Usage Time (mins)', 'State', 'Region', 'Speciality', 'Count of Survey Attempts']
        df['Predicted'] = self.model.predict(df[features_for_prediction])

        # Get NPIs of active doctors
        recommended_doctors = df[df['Predicted'] == 1][['NPI']]
        return recommended_doctors