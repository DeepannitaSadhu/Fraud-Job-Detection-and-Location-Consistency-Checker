import streamlit as st
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Job Fraud Predictor",
    page_icon="🔍"
)

st.title("🔍 Job Fraud Predictor")
st.markdown(
     """
    This application helps predict the likelihood of a job posting being fraudulent.
    Fill in the details below and click 'Predict' to see the result.
    
    **Note:** This is a conceptual app. The prediction logic is a simplified simulation
    based on common red flags. This project is based on a particular dataset, not a real-world application.
    """
)

try:
    with open("trained_model.sav", "rb") as file:
        model_lr, tfidf_vectorizer = pickle.load(file)
    st.success("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file 'trained_model.sav' not found. Please ensure the model is trained and saved.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.header("Job Posting Details")

job_locations = [
    "New York, NY", "San Francisco, CA", "London, UK", "Remote",
    "Singapore", "Tokyo, Japan", "Sydney, Australia", "Berlin, Germany", "Other"
]
employment_types = [
    "Full-time", "Part-time", "Contract", "Temporary", "Other"
]

company_name = st.text_area("Company Name", placeholder="Enter the company name")
location = st.selectbox("Job Location", options=["Select Location"] + job_locations)
description = st.text_area("Job Description", placeholder="Enter job description")
requirements = st.text_area("Job Requirements", placeholder="Enter job requirements")
employment_type = st.selectbox("Employment Type", options=["Select Type"] + employment_types)
company_profile = st.text_area("Company Profile", placeholder="Enter company profile")
benefits = st.text_area("Benefits", placeholder="Enter job benefits")
telecommuting = st.checkbox("Telecommuting")
has_company_logo = st.checkbox("Has company logo")
has_questions = st.checkbox("Has screening questions")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)         # remove links
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

text_combined = f"{company_name} {location} {company_profile} {description} {requirements} {employment_type} {benefits}"

risky_locations = {
    "remote", "us", "united states", "india",
    "uk", "london", "canada", "austin, tx",
    "chicago, il", "new york, ny"
}

def check_location_consistency(loc):
    loc = str(loc).strip().lower()
    if not loc or loc == "nan" or loc == "":
        return 0
    if loc in risky_locations:
        return 0   
    return 1       

def predict_fraud(text_combined, location, telecommuting, has_company_logo, has_questions):
    cleaned_text = clean_text(text_combined)

    location_consistent = check_location_consistency(location)

    X_text = tfidf_vectorizer.transform([cleaned_text]).toarray()

    X_numerical = np.array([
        telecommuting,
        has_company_logo,
        has_questions,
        location_consistent
    ]).reshape(1, -1)

    X_combined = np.concatenate([X_text, X_numerical], axis=1)

    is_fraudulent = model_lr.predict(X_combined)[0]
    proba = model_lr.predict_proba(X_combined)[0]
    confidence_score = proba[1] if is_fraudulent == 1 else proba[0]

    return is_fraudulent, confidence_score, location_consistent


if st.button("Predict"):
    if location == "Select Location":
        st.warning("⚠ Please select a job location.")
    elif not any([company_name, company_profile, description, requirements, benefits]):
        st.warning("⚠ Please fill in at least one text field.")
    else:
        with st.spinner("🔎 Analyzing job posting..."):
            is_fraud, confidence, location_consistent = predict_fraud(
                text_combined, location, telecommuting, has_company_logo, has_questions
            )

        st.subheader("Prediction Result")

        if is_fraud:
            st.error("❌ This job posting is likely fraudulent.")
            st.markdown(f"Confidence score: **{confidence:.2f}**")
        else:
            st.success("✅ This job posting appears legitimate.")
            st.markdown(f"Confidence score: **{confidence:.2f}**")

        st.subheader("📍 Location Consistency Check")
        if location_consistent:
            st.success("✅ The job location looks safe (not a top risky location).")
        else:
            st.warning("⚠ The job location is in the list of top risky fake-job locations.")

        st.markdown(
            """
            ⚠ **Note:** Always be cautious and do your own research before applying for any job.
            """
        )
