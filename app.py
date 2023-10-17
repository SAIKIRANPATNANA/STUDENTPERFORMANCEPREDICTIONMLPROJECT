import streamlit as st
from src.StudentPerformancePredictionProject.pipeline.prediction import CustomData,PredictionPipeline
import pandas as pd
def main():
    st.title("Student Performance Indicator")
    # Collect User Input
    gender = st.selectbox("Gender", ["", "male", "female"])
    race = st.selectbox("Race", ["", "group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education",
        ["", "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
    lunch = st.selectbox("Lunch Type", ["", "free/reduced", "standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["", "none", "completed"])
    reading_score = st.slider("Reading Score out of 100", 0, 100, 50)
    writing_score = st.slider("Writing Score out of 100", 0, 100, 50)

    if st.button("Predict your Maths Score"):
        data = CustomData(
            gender=gender,
            race=race,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        st.subheader(f"The prediction is: {round(results[0])}")

if __name__ == "__main__":
    main()
