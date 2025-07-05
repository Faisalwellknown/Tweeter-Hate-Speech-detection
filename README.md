# Tweeter-Hate-Speech-detection
Faisal Shaikh project for detecting hate speech 

# Twitter Hate Speech Detection using BERT

This project implements a machine learning model using BERT (Bidirectional Encoder Representations from Transformers) to detect hate speech in tweets. The model is trained on a labeled dataset and provides a classification of tweets into hate, offensive, or neutral categories.

## 🔍 Project Objective

The goal is to:
- Analyze tweets for hateful or offensive content
- Build a robust NLP pipeline using the power of BERT
- Evaluate model performance on real-world social media data

## 🧠 Technologies Used

- Python 🐍
- Jupyter Notebook 📒
- TensorFlow & Keras
- HuggingFace Transformers
- Pandas, NumPy, Matplotlib
- BERT Pretrained Model

## 📁 Project Structure

├── Twitter_Hate_Speech_Detection_using_BERT.ipynb
├── labeled_data.csv
├── saved_model/
│ ├── assets/
│ ├── variables/ (model weights not included due to 100MB limit)
│ └── keras_model.pb
└── README.md



## 📊 Dataset

We use the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/aymanarafa/twitter-hate-speech) which includes:
- **Tweet text**
- **Labels** for: hate speech, offensive language, and neither

The file `labeled_data.csv` contains all training data.

## 🧠 Model Architecture

- Tokenizer: BERT base uncased
- Preprocessing: lowercasing, padding, truncation
- Fine-tuned using TensorFlow/Keras
- Output Layer: Softmax (3 classes)

## 🚀 How to Run

> ⚠️ The trained model file `variables.data-00000-of-00001` was removed from the repo (GitHub size limit). Please download it manually (see below).

### 1. Clone the Repository

```bash
git clone https://github.com/Faisalwellknown/Tweeter-Hate-Speech-detection.git
cd Tweeter-Hate-Speech-detection
2. Install Dependencies

pip install -r requirements.txt
3. Download the Model Weights
Download saved_model weights from Google Drive

Place the file inside:

saved_model/variables/variables.data-00000-of-00001
4. Run the Notebook
Use Jupyter or Colab:


jupyter notebook Twitter_Hate_Speech_Detection_using_BERT.ipynb
📈 Results
Accuracy: ~90%

F1-Score: Varies across classes

Confusion Matrix & classification report included in the notebook

📚 References
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al.)

Kaggle Hate Speech Dataset

HuggingFace Transformers

👤 Author
Faisal Shaikh
GitHub: @Faisalwellknown
Project: Hate Speech Detection using BERT

💡 Future Improvements
Use DistilBERT or RoBERTa for faster training

Deploy as a web app with Flask or Streamlit

Real-time tweet classification API
Accuracy:
![Screenshot 2025-07-05 152828](https://github.com/user-attachments/assets/923e5310-41cc-4b8d-80b0-3371698eea03)
Output:
![Screenshot 2025-07-05 152858](https://github.com/user-attachments/assets/7e4bcc31-caed-4247-b73a-bc58aedc06f1)


