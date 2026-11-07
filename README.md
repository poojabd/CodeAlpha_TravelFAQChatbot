# ✈️ TravelMate – Travel FAQ Chatbot

## About the Project

TravelMate is a Python-based Travel FAQ Chatbot that helps users find answers to common travel-related questions.

**`README.md`**.

Users can ask questions about flights, hotels, baggage, visas, airport procedures and other travel essentials. The chatbot uses Natural Language Processing (NLP) to identify the most relevant FAQ and return its answer.

## Features

- Simple and user-friendly interface.
- Answers 25 predefined travel-related FAQs.
- NLP-based text preprocessing.
- FAQ matching using TF-IDF.
- Cosine Similarity to find the closest matching question.
- Provides a fallback response when no suitable answer is found.
- Interactive chatbot interface.

## Technologies Used

- **Python:** Main programming language.
- **Streamlit:** Web application interface.
- **NLTK:** Text tokenization and stemming.
- **Scikit-learn:** TF-IDF vectorization and Cosine Similarity.
- **HTML and CSS:** Website styling.

## How It Works

1. The user enters a travel-related question.
2. The question is converted into lowercase text and tokenized.
3. Stop words are removed and stemming is performed using NLTK.
4. TF-IDF converts the stored FAQ questions into numerical vectors.
5. Cosine Similarity compares the user's question with the FAQ questions.
6. The chatbot returns the answer associated with the closest matching question.
7. If no suitable match is found, a fallback message is displayed.

## Project Structure

```text
CodeAlpha_TravelFAQChatbot/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd CodeAlpha_TravelFAQChatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Future Improvements

- Expand the travel FAQ dataset.
- Improve question-matching accuracy.
- Add multilingual support.
- Introduce additional chatbot features.

## Project Details

**Project:** TravelMate – Travel FAQ Chatbot

**Category:** Artificial Intelligence / Natural Language Processing

**Developed Using:** Python, Streamlit, NLTK and Scikit-learn
