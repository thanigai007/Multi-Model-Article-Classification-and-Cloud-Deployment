import streamlit as st
import mysql.connector
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from wordcloud import WordCloud

# ------------------ Load dataset ------------------
train_df = pd.read_csv("D:/Project/Guvi_Project/Final Project/dataset/train.csv")
test_df = pd.read_csv("D:/Project/Guvi_Project/Final Project/dataset/test.csv")

# Map labels
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
train_df['Class Index'] = train_df['Class Index'].map(label_map)
test_df['Class Index'] = test_df['Class Index'].map(label_map)

# ------------------ MySQL connection ------------------
def get_connection():
    return mysql.connector.connect(
        host="articledb.c6386ycuirkz.us-east-1.rds.amazonaws.com",
        user="admin",
        password="thanigai1234",
        database="user_db"
    )

# ------------------ Hashing ------------------
def make_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ Signup ------------------
def signup(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        conn.close()
        return False
    hashed_pw = make_hash(password)
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
    conn.commit()
    conn.close()
    return True

# ------------------ Login ------------------
def login(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_pw = make_hash(password)
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, hashed_pw))
    user = cursor.fetchone()
    conn.close()
    return user

# ------------------ APP ------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if not st.session_state.logged_in:
        # -------- LOGIN / SIGNUP --------
        st.title("🔐 Login / Signup")

        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                user = login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        elif choice == "Sign Up":
            new_user = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if signup(new_user, new_password):
                    st.success("✅ Account created successfully! Please log in.")
                else:
                    st.error("⚠️ Username already exists.")
    else:
        # -------- AFTER LOGIN --------
        st.sidebar.success(f"Welcome {st.session_state.username}👋")
        
        page = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Prediction"])
        # 🚀 Logout Button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

        if page == "Introduction":
            st.header("📘 Introduction")
            st.write("""
            This project classifies news articles into **4 categories**:  
            1. World 🌍  
            2. Sports 🏅  
            3. Business 💼  
            4. Sci/Tech 🔬  

            It uses **ML, Deep Learning, and Transformers** for predictions.  
            """)

        elif page == "EDA":
            st.header("📊 Exploratory Data Analysis")
            eda_option = st.selectbox(
                "Choose EDA visualization:",
                [
                    "Class Distribution",
                    "Word Clouds (per category)",
                    "Average Word Count & Title/Description Length",
                    "Heatmap of Frequent Words per Class"
                ]
            )

            if eda_option == "Class Distribution":
                plt.figure(figsize=(8, 5))
                sns.countplot(data=train_df, x='Class Index', order=train_df['Class Index'].value_counts().index, palette="Set2")
                plt.title("Class Distribution")
                st.pyplot(plt)

            elif eda_option == "Word Clouds (per category)":
                categories = train_df['Class Index'].unique()
                for cat in categories:
                    st.write(f"### {cat}")
                    text = " ".join(train_df[train_df['Class Index'] == cat]['Description'].astype(str))
                    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)

            elif eda_option == "Average Word Count & Title/Description Length":
                st.subheader("Word Count & Text Length Analysis")
                train_df['word_count'] = train_df['Description'].apply(lambda x: len(str(x).split()))
                plt.figure(figsize=(8, 5))
                sns.histplot(train_df['word_count'], bins=50, kde=True)
                plt.title("Distribution of Word Counts")
                st.pyplot(plt)

            elif eda_option == "Heatmap of Frequent Words per Class":
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(stop_words="english", max_features=20)
                X_counts = vectorizer.fit_transform(train_df['Description'])
                df_freq = pd.DataFrame(X_counts.toarray(), columns=vectorizer.get_feature_names_out())
                df_freq['Class Index'] = train_df['Class Index']
                heatmap_data = df_freq.groupby("Class Index").sum()
                sns.heatmap(heatmap_data.T, cmap="YlGnBu", annot=True, fmt="d")
                st.pyplot(plt)

        elif page == "Prediction":
            st.header("🔮 Prediction")
            article = st.text_area("Enter article text:")
            model_choice = st.selectbox("Choose model", ["ML", "DL", "Transformer"])

            if st.button("Predict"):
                if not article.strip():
                    st.warning("Please enter text")
                else:
                    if model_choice == "ML":
                        ml_model = joblib.load("D:/Project/Guvi_Project/Final Project/Saved Models/ML/Logistic Regression.pkl")
                        vectorizer = joblib.load("D:/Project/Guvi_Project/Final Project/Saved Models/ML/ml_vectorizer.pkl")
                        X_vec = vectorizer.transform([article])
                        pred = ml_model.predict(X_vec)[0]
                        conf = np.max(ml_model.predict_proba(X_vec))
                    elif model_choice == "DL":
                        dl_model = load_model("D:/Project/Guvi_Project/Final Project/Saved Models/DL/CNN_model.h5")
                        tokenizer = joblib.load("D:/Project/Guvi_Project/Final Project/Saved Models/DL/dl_tokenizer.pkl")
                        seq = tokenizer.texts_to_sequences([article])
                        pad_seq = pad_sequences(seq, maxlen=200, padding="post")
                        probs = dl_model.predict(pad_seq)
                        pred = np.argmax(probs) + 1
                        conf = np.max(probs)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                        model = TFAutoModelForSequenceClassification.from_pretrained(
                            "D:/Project/Guvi_Project/Final Project/Saved Models/TF"
                        )
                        inputs = tokenizer(article, return_tensors="tf", truncation=True, padding=True, max_length=200)
                        outputs = model(inputs)
                        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()
                        pred = np.argmax(probs) + 1
                        conf = np.max(probs)

                    st.success(f"Predicted Category: {pred} - {label_map[pred]}")
                    st.info(f"Confidence Score: {conf:.4f}")

if __name__ == "__main__":
    main()
