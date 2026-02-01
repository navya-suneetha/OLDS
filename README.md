# Telugu-English Code-Mixed Offensive Language Detection

This project is a **Streamlit web application** for detecting hate speech in Telugu-English code-mixed text using a **Support Vector Machine (SVM)** classifier with **TF-IDF** features.

The app automatically:
* Loads datasets
* Preprocesses text
* Trains an SVM model
* Evaluates performance
* Allows live user predictions

---

## 🚀 Demo Features

* 📂 **Automatic dataset loading** from local folder
* 🧹 **Text preprocessing** (cleaning & normalization)
* 🧠 **Machine learning model** (TF-IDF + Linear SVM)
* 📊 **Accuracy & classification report** display
* 📝 **Live hate speech prediction** for user input

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Handling:** Pandas
* **ML Library:** Scikit-learn
* **Processing:** Regex
* **Algorithm:** SVM (LinearSVC)
* **Vectorization:** TF-IDF Vectorizer

---

## 📁 Project Structure

```text
.
├── app.py                 # Main Streamlit application code
├── dataset/               # Folder for CSV data
│   ├── D5_train_data.csv
│   └── D5_test_data.csv
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation
```

## 📊 Dataset Format

Both training and testing CSV files must contain the following columns:

| Column Name | Description |
| :--- | :--- |
| **Comments** | Telugu-English code-mixed text |
| **Label** | Hate or Non-Hate |

## ⚙️ Model Pipeline

The model is built using a **Scikit-learn Pipeline** to ensure that raw text is consistently transformed into numerical features before being passed to the classifier.


### 1. TF-IDF Vectorizer
This step converts text into a matrix of **TF-IDF (Term Frequency-Inverse Document Frequency)** features, focusing on the importance of words relative to the entire dataset.

* **ngram range:** `(1, 2)` — Captures both individual words and two-word combinations (bigrams) to better understand context.
* **max features:** `25,000` — Restricts the vocabulary to the top 25,000 most frequent terms to optimize performance.
* **sublinear TF scaling:** `Enabled` — Applies log scaling to term frequencies to dampen the influence of very frequent words.


### 2. Linear Support Vector Machine (LinearSVC)
The vectorized text is then fed into a **LinearSVC** model, which is highly effective for high-dimensional text classification tasks.

* **Kernel:** Linear.
* **Task:** Binary Classification (**"Hate"** vs. **"Non-Hate"**).


---

## 🧹 Preprocessing Steps

To ensure high-quality predictions, the input text undergoes an essential cleaning process via a custom function:

* **Data Type Enforcement:** Converts all inputs (including potential numbers or data errors) into string format.
* **Handle Missing Data:** Safely detects and converts `NaN` or missing values to empty strings to prevent training/prediction crashes.
* **Whitespace Normalization:** Uses Regular Expressions (`re.sub`) to collapse multiple spaces into a single space and strips padding from the ends.
* **Structure Preservation:** Intentionally avoids aggressive filtering (like removing special characters or punctuation) to keep the original **Telugu-English code-mixed** nuances intact for the classifier.

## ▶️ How to Run Locally

Follow these steps to get the application running on your local machine:

### 1️⃣ Install Dependencies
Open your terminal (or VS Code terminal) and run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit app
```bash
streamlit run app.py
```
## 📝 Usage

Follow these steps to use the application interface:

1.  **Initialize the Model:** Click the **🚀 Train SVM Model** button to load the dataset and begin the training process.
2.  **Evaluate Performance:** Once training is complete, the app will automatically display the **Accuracy Score** and a detailed **Classification Report** (Precision, Recall, and F1-Score).
3.  **Input Text:** Locate the text area and enter any **Telugu-English mixed comment** you wish to analyze.
4.  **Get Prediction:** Click the **Predict** button.
5.  **View Results:** The system will process the text and display either a **✅ Non-Offensive Language** or a **🚫 Offensive Language Detected** result instantly based on the model's analysis.
---
