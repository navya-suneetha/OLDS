import streamlit as st
import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------
# PATH HANDLING
# ---------------------------------------------------

working_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(working_dir, "dataset")

TRAIN_FILE = "D5_train_data.csv"
TEST_FILE = "D5_test_data.csv"

train_path = os.path.join(DATASET_PATH, TRAIN_FILE)
test_path = os.path.join(DATASET_PATH, TEST_FILE)

# ---------------------------------------------------
# PREPROCESSING FUNCTION
# ---------------------------------------------------

def pre_processing(X):

    processed_tweets = []

    for tweet in range(0, len(X)):

        # Convert NaN to empty string safely
        text = str(X[tweet]) if pd.notna(X[tweet]) else ""

        # Remove extra spaces only
        text = re.sub(r"\s+", " ", text).strip()

        processed_tweets.append(text)

    return processed_tweets



# ---------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------

st.set_page_config(
    page_title="Telugu-English Offensive Language Detection",
    layout="centered"
)

st.title("🔥 Telugu-English Code-Mixed Offensive Language Detection (SVM)")
st.write("Automatic dataset loading + preprocessing + live prediction")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

if not os.path.exists(train_path) or not os.path.exists(test_path):
    st.error("❌ Dataset files not found inside dataset folder!")
    st.stop()

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

st.success("Datasets loaded successfully!")

# ---------------------------------------------------
# APPLY PREPROCESSING
# ---------------------------------------------------

X_train = pre_processing(train_df["Comments"].values)
y_train = train_df["Label"]

X_test = pre_processing(test_df["Comments"].values)
y_test = test_df["Label"]

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

if "model" not in st.session_state:

    st.info("Click below to train the SVM model.")

    if st.button("🚀 Train SVM Model"):

        with st.spinner("Training in progress..."):

            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=25000,
                    sublinear_tf=True
                )),
                ("svm", LinearSVC())
            ])

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.session_state["model"] = model

        st.success("✅ Model trained successfully!")

        st.subheader("📊 Test Accuracy")
        st.write(f"**{acc:.4f}**")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

else:
    st.success("✅ Model already trained!")

# ---------------------------------------------------
# PREDICTION UI
# ---------------------------------------------------

st.divider()
st.header("📝 Test Your Own Comment")

user_text = st.text_area(
    "Enter Telugu-English mixed sentence:",
    placeholder="nee behaviour chala worst ra..."
)

if st.button("Predict"):

    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = pre_processing([user_text])[0]

        pred = st.session_state["model"].predict([cleaned])[0]

        if pred == "Hate":
            st.error("🚫 Prediction: HATE SPEECH")
        else:
            st.success("✅ Prediction: NON-HATE")
