# Multi-Model-Article-Classification-and-Cloud-Deployment

            ### 📝 Problem Statement  
            With the growing volume of online text data, classifying documents/articles 
            into the right categories is essential for organizing, retrieving, and 
            analyzing information. Manual classification is time-consuming and 
            error-prone, which motivates the need for automated text classification 
            using **Machine Learning (ML)**, **Deep Learning (DL)**, and **Transformer models**.  

            ---
            ### 🎯 Solution
            This project develops a **multi-model text classification system** that 
                leverages:
                - **ML Models** (e.g., Logistic Regression, Naive Bayes, SVM) with TF-IDF features.  
                - **DL Models** (e.g., LSTMs, CNNs) with word embeddings.  
                - **Transformers** (e.g., BERT, DistilBERT, RoBERTa) for contextual embeddings.  

                The models are trained and compared for performance, then deployed 
                in a **Streamlit application** on **AWS EC2** (with optional Hugging Face Spaces deployment).  

            ---

            ### 📂 Data Description  
            - The dataset contains **news articles** with text content and their corresponding category labels.  
            - Each record consists of:  
            - **Title / Content**: The article text to classify  
            - **Category Label**: One of the predefined categories  
            - Preprocessing includes tokenization, cleaning, and vectorization for ML/DL models.  

            ---

            ### 🗂️ Categories  
            The system classifies articles into the following categories:  
            - 🌍 **World**  
            - 💼 **Business**  
            - 🏅 **Sports**  
            - 💻 **Technology**  

            ---

            ### 🛠️ Tools & Technologies Used  
            - **Programming Language**: Python  
              - **Framework**: Streamlit (for web app UI)  
              - **Models**:  
              - Machine Learning (Logistic Regression, Naive Bayes, SVM, etc.)  
              - Deep Learning (CNN/BiLSTM)  
              - Transformer Models (DistilBERT via Hugging Face)  
            - **Cloud Infrastructure**:  
              - AWS EC2 (hosting Streamlit app)  
              - AWS S3 (storing model files if needed)  
              - AWS RDS (storing user login and activity logs)  
            - **Libraries**: scikit-learn, TensorFlow/Keras, Hugging Face Transformers, Pandas, NumPy 
