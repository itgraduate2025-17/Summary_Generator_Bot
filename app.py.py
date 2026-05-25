import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

# ---------------- FAQ DATA ----------------
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return products within 7 days of delivery."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking ID sent to your email."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to most countries worldwide."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact support via email or live chat."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, PayPal, and bank transfer."
    }
]

questions = [f["question"] for f in faqs]
answers = [f["answer"] for f in faqs]

# ---------------- TEXT CLEANING ----------------
def clean(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

clean_questions = [clean(q) for q in questions]

# ---------------- VECTOR MODEL ----------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(clean_questions)

# ---------------- RESPONSE FUNCTION ----------------
def get_response(user_input):

    user_input = clean(user_input)

    user_vec = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()

    if similarity[0][index] < 0.2:
        return "Sorry, I could not understand your question."

    return answers[index]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Support Chatbot",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.user-msg {
    background-color: #d1e7ff;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
}

.bot-msg {
    background-color: #eeeeee;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #1f77b4;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<div class='title'>Customer Support Chatbot</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Ask questions about orders, shipping, payments, returns, and support.</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("About Chatbot")

st.sidebar.info("""
This chatbot is designed for e-commerce and business websites.

It helps customers with:
- Orders
- Shipping
- Payments
- Return Policy
- Customer Support
""")

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- USER INPUT ----------------
user_input = st.text_input(
    "Enter your question"
)

col1, col2 = st.columns([1, 1])

with col1:
    send = st.button("Send")

with col2:
    clear = st.button("Clear Chat")

# ---------------- SEND MESSAGE ----------------
if send and user_input:

    response = get_response(user_input)

    st.session_state.chat.append(("You", user_input))

    st.session_state.chat.append(("Bot", response))

# ---------------- CLEAR CHAT ----------------
if clear:
    st.session_state.chat = []

# ---------------- DISPLAY CHAT ----------------
for sender, msg in st.session_state.chat:

    if sender == "You":

        st.markdown(
            f"<div class='user-msg'><b>You:</b> {msg}</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"<div class='bot-msg'><b>Bot:</b> {msg}</div>",
            unsafe_allow_html=True
        )