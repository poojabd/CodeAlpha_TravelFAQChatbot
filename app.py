
import streamlit as st
import html
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TravelMate",
    page_icon="✈️",
    layout="centered"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

/* WHITE WEBSITE BACKGROUND */

.stApp {
    background-color: #FFFFFF;
    color: #000000;
}

header[data-testid="stHeader"] {
    background-color: #FFFFFF;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HEADER */

.travel-header {
    background-color: #397FB8;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 28px;
}

.travel-header h1 {
    color: #FFFFFF !important;
    font-size: 38px;
    margin-bottom: 5px;
}

.travel-header p {
    color: #FFFFFF !important;
    font-size: 16px;
    margin: 0;
}

/* WELCOME SECTION */

.welcome {
    color: #000000;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
}

.subtitle {
    color: #000000;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 22px;
}

/* QUESTION INPUT BOX */

div[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #397FB8 !important;
    border: 2px solid #397FB8 !important;
    border-radius: 10px !important;
}

div[data-testid="stTextInput"] input {
    background-color: #397FB8 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    font-size: 16px !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

/* INPUT FOCUS */

div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border: 2px solid #245F91 !important;
    box-shadow: 0 0 0 2px rgba(57, 127, 184, 0.15) !important;
}

/* ASK BUTTON AND SIDEBAR BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #397FB8 !important;
    color: #FFFFFF !important;
    border: 1px solid #397FB8 !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #286A9F !important;
    border-color: #286A9F !important;
    color: #FFFFFF !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

.sidebar-title {
    color: #397FB8;
    font-size: 23px;
    font-weight: bold;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h3 {
    color: #000000;
}

/* CHAT QUESTION AND ANSWER BOXES */

.question-box {
    background-color: #397FB8;
    color: #FFFFFF;
    padding: 15px 18px;
    border-radius: 12px;
    margin-top: 18px;
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.answer-box {
    background-color: #397FB8;
    color: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
    font-size: 15px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.answer-title {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: bold;
    margin-bottom: 8px;
}

.question-box p,
.answer-box p {
    color: #FFFFFF !important;
}

/* HIDE STREAMLIT FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- NLP PREPROCESSING ----------------

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
stemmer = PorterStemmer()

stop_words = {
    "the", "is", "are", "am", "a", "an", "and", "or",
    "to", "for", "of", "in", "on", "at", "with", "my",
    "i", "me", "you", "your", "it", "do", "does", "can",
    "what", "how", "when", "where", "which", "if", "should"
}


def preprocess_text(text):
    text = text.lower()
    words = tokenizer.tokenize(text)

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------- TRAVEL FAQ DATASET ----------------

faqs = [
    {
        "question": "What documents are required for travelling?",
        "answer": "You generally need a valid government-issued ID for domestic travel. International travel usually requires a valid passport and, depending on your destination, a visa."
    },
    {
        "question": "How early should I reach the airport?",
        "answer": "For domestic flights, try to reach the airport at least 2 hours before departure. For international flights, arrive around 3 hours early."
    },
    {
        "question": "Can I cancel my flight ticket?",
        "answer": "Yes, most airlines allow ticket cancellation. Cancellation charges and refunds depend on the airline, ticket type and time of cancellation."
    },
    {
        "question": "Do I need a visa to travel abroad?",
        "answer": "Visa requirements depend on your destination and nationality. Check the official embassy or immigration website before travelling."
    },
    {
        "question": "How can I book a flight ticket?",
        "answer": "You can book flight tickets through airline websites or trusted travel booking platforms. Compare prices and check the baggage and cancellation rules before paying."
    },
    {
        "question": "What should I pack for a trip?",
        "answer": "Pack clothes, toiletries, travel documents, medicines, chargers and other essentials. Check the weather and baggage allowance before packing."
    },
    {
        "question": "What items are not allowed in cabin baggage?",
        "answer": "Sharp objects, certain tools and liquids exceeding the permitted limit are generally restricted in cabin baggage. Check your airline's baggage rules."
    },
    {
        "question": "What happens if my flight is delayed?",
        "answer": "Check the airline's updates for the latest departure time. Contact airline staff for information about rebooking, refunds or assistance."
    },
    {
        "question": "How can I check my flight status?",
        "answer": "You can check your flight status using the airline's official website or mobile application with your flight number."
    },
    {
        "question": "Can I change my travel date?",
        "answer": "Most airlines allow date changes depending on ticket conditions. Additional charges or fare differences may apply."
    },
    {
        "question": "Is travel insurance necessary?",
        "answer": "Travel insurance is useful because it may cover medical emergencies, trip cancellations, lost baggage and other unexpected situations."
    },
    {
        "question": "How can I check in online?",
        "answer": "Visit your airline's website or app, enter your booking details and follow the check-in instructions. Online check-in availability varies by airline."
    },
    {
        "question": "Can I travel with my pet?",
        "answer": "Some airlines allow pets under specific conditions. Check the airline's pet policy and required documents before travelling."
    },
    {
        "question": "What should I do if I lose my passport?",
        "answer": "Report the loss to local police and contact your country's nearest embassy or consulate for guidance on obtaining an emergency travel document."
    },
    {
        "question": "How can I find cheap flight tickets?",
        "answer": "Compare prices across booking platforms, travel during less busy periods and book in advance when possible. Prices can change frequently."
    },
    {
        "question": "What is a boarding pass?",
        "answer": "A boarding pass is a document issued after check-in. It contains your flight details, gate information and boarding time."
    },
    {
        "question": "Can I carry food on a flight?",
        "answer": "Rules depend on the airline and destination. Solid food may be permitted, but liquids and certain food items can have restrictions."
    },
    {
        "question": "What is the airport boarding process?",
        "answer": "The process generally includes check-in, security screening, reaching your boarding gate and presenting your boarding pass when boarding begins."
    },
    {
        "question": "What is the difference between domestic and international flights?",
        "answer": "Domestic flights travel within one country, while international flights travel between different countries and generally require additional travel documents."
    },
    {
        "question": "How can I book a hotel?",
        "answer": "You can book hotels through hotel websites or trusted booking platforms. Compare location, reviews, facilities and cancellation policies."
    },
    {
        "question": "When is the best time to travel?",
        "answer": "The best time depends on your destination, weather, budget and travel preferences. Research seasonal conditions before planning."
    },
    {
        "question": "What should I do if my luggage is lost?",
        "answer": "Report missing luggage to the airline's baggage service desk at the airport. Keep your baggage receipt and obtain a reference number for tracking."
    },
    {
        "question": "Can I use my mobile phone during a flight?",
        "answer": "You can generally use your phone in airplane mode. Follow the airline crew's instructions regarding electronic devices."
    },
    {
        "question": "How can I stay safe while travelling?",
        "answer": "Keep your belongings secure, use trusted transport, carry copies of important documents and stay aware of your surroundings."
    },
    {
        "question": "What is the baggage allowance for flights?",
        "answer": "Baggage allowance depends on the airline, route and ticket type. Check your booking details for cabin and checked baggage limits."
    }
]


# ---------------- TF-IDF MODEL ----------------

faq_questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)


def get_answer(user_question):

    processed_question = preprocess_text(user_question)

    if not processed_question.strip():
        return "Please enter a travel-related question."

    question_vector = vectorizer.transform([processed_question])

    similarity = cosine_similarity(
        question_vector,
        faq_vectors
    )

    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    if best_score >= 0.20:
        return faqs[best_match]["answer"]

    return (
        "Sorry, I couldn't find a suitable answer to your question. "
        "Please try asking something related to travel, flights, "
        "hotels or baggage."
    )


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">✈ TravelMate</div>',
        unsafe_allow_html=True
    )

    st.write("Your personal travel FAQ assistant.")

    st.markdown("---")

    st.markdown("### Suggested Questions")

    suggestions = [
        "How early should I reach the airport?",
        "Do I need a visa to travel abroad?",
        "How can I book a flight ticket?",
        "What happens if my luggage is lost?",
        "What should I pack for a trip?"
    ]

    for question in suggestions:

        if st.button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "Ask questions about flights, hotels, visas, "
        "baggage and other travel essentials."
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="travel-header">
    <h1>✈ TravelMate</h1>
    <p>Your Smart Travel FAQ Assistant</p>
</div>
""", unsafe_allow_html=True)


# ---------------- WELCOME SECTION ----------------

st.markdown(
    '<div class="welcome">Hello, Traveller! 👋</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Planning a trip? Ask me anything about flights, hotels, '
    'baggage, visas and travel essentials.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- QUESTION INPUT ----------------

with st.form("travel_question_form", clear_on_submit=True):

    user_query = st.text_input(
        "Ask your question",
        placeholder="Type your travel question here...",
        label_visibility="collapsed"
    )

    submit_button = st.form_submit_button(
        "Ask TravelMate",
        use_container_width=True
    )


# ---------------- PROCESS QUESTION ----------------

if st.session_state.pending_question:

    user_query = st.session_state.pending_question
    st.session_state.pending_question = None
    submit_button = True


if submit_button and user_query and user_query.strip():

    answer = get_answer(user_query)

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()


# ---------------- DISPLAY CONVERSATION ----------------

for message in st.session_state.messages:

    safe_message = html.escape(message["content"]).replace("\n", "<br>")

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="question-box">
                <b>Your Question</b><br><br>
                {safe_message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-title">✈ TravelMate Answer</div>
                {safe_message}
            </div>
            """,
            unsafe_allow_html=True
        )

