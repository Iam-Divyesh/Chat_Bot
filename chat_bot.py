import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pdfplumber
import numpy as np
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="HeadsIn Chatbot", page_icon="🤖")
load_dotenv()

# Azure OpenAI Setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint="https://job-recruiting-bot.openai.azure.com/"
)
chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Utility Functions
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def chunk_text(text, chunk_size=700):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def scrape_page(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        return clean_text(soup.get_text(separator=' ', strip=True))
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# Cache embedding preparation
@st.cache_resource
def prepare_embeddings():
    urls = [
        "https://headsin.co/",
        "https://headsin.co/code-of-conduct",
        "https://headsin.co/privacy-policy",
        "https://headsin.co/terms-and-conditions",
        "https://headsin.co/about-us",
        "https://headsin.co/contact-us",
        "https://headsin.co/build-resume-page",
        "https://headsin.co/candidate",
        "https://headsin.co/company"
        # "https://headsin.co/discover/ai-driven-job-matching-2025",
        # "https://headsin.co/discover/frustration-of-job-seekers-in-india",
        # "https://headsin.co/discover/why-smbs-in-india-need-smart-hiring-platform",
        # "https://headsin.co/discover/culture-fit-hiring-in-india",
        # "https://headsin.co/discover/ai-resume-screening-india",
        # "https://headsin.co/discover/gig-economy-jobs-india",
        # "https://headsin.co/discover/hybrid-jobs-in-india",
        # "https://headsin.co/discover/emerging-job-sectors-india-2025"
    ]

    web_text = "\n".join([scrape_page(url) for url in urls])

    with pdfplumber.open("./assets/HeadsIn_Public_Chatbot_Report_2025.pdf") as pdf:
        pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    combined = web_text + "\n" + pdf_text
    chunks = [clean_text(c) for c in chunk_text(combined, chunk_size=700) if c.strip()]

    embeddings = []
    for idx, chunk in enumerate(chunks):
        resp = client.embeddings.create(model=embedding_deployment, input=[chunk])
        embeddings.append({"text": chunk, "embedding": resp.data[0].embedding})
    return embeddings

# Load all embeddings
chunk_embeddings = prepare_embeddings()

FINAL_BLOCK_MESSAGE = (
    "Thanks for checking with us. Feel free to visit our website.\n\n"
    "If you're seeking a job, visit: https://headsin.co/auth  \n\n"
    "If you're looking to hire candidates, go to: https://company.headsin.co/auth \n\n"
    "For Further Details, Contact: \n\n"
    "Call : +91 97734 - 97763 \n\n"
    "Email : info@headsin.co \n"
)


# UI Setup
st.title("🤖 HeadsIn's Chatbot")

# Init session state
if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hello! How can I help you today? 😊")]
if "irrelevant_count" not in st.session_state:
    st.session_state.irrelevant_count = 0

# Disable chat if 2+ irrelevant
if st.session_state.irrelevant_count >= 2:
    with st.chat_message("assistant"):
        st.write(FINAL_BLOCK_MESSAGE)
    st.stop()  # stops further execution
else:
    user_input = st.chat_input("Ask me anything...")

# Handle input
# Handle input
if user_input:
    cleaned_input = clean_text(user_input)

    # Get embedding for user input
    user_emb = client.embeddings.create(model=embedding_deployment, input=[cleaned_input])
    user_vec = user_emb.data[0].embedding

    # Find best matching chunk
    best_chunk = max(chunk_embeddings, key=lambda c: cosine_similarity(user_vec, c["embedding"]))["text"]

    # Full prompt with relevant context
    system_prompt = (
        "You are a professional, friendly, and concise AI assistant for HeadsIn, an AI-powered job search and hiring platform based in India.\n\n"

        "Guidelines:\n"
        "- ALWAYS reply in 1–2 short sentences. Do not write paragraphs.\n"
        "- You help users with job search, hiring, resumes, assessments, interview process, and HeadsIn related questions.\n"
        "- Greet users warmly when they say 'hi', 'hello', or introduce themselves (e.g., 'I am Renish').\n"
        "- If a user is looking for a job (e.g., role, location), guide them on how to apply via HeadsIn.\n"
        "- If a recruiter wants to hire, guide them to post a job using the platform.\n"
        "- If a question seems unrelated, respond once politely, and stop after 2 unrelated messages.\n"

        "Special Instructions:\n"
        "- If asked: 'What is HeadsIn?'\n"
        "  → HeadsIn is an AI-powered job platform that matches candidates to jobs and helps recruiters hire faster.\n"
        "- If asked: 'How do I apply for a job?'\n"
        "  → Go to https://headsin.co, log in, create your resume, complete a short assessment, and apply.\n"
        "- If asked: 'How do I post a job?'\n"
        "  → Visit https://company.headsin.co, log in as a recruiter, fill job details, and post.\n"
        "- If asked: 'In how many days will I get a job?'\n"
        "  → It depends on your profile and company requirements.\n"
        "- If asked about support:\n"
        "  → Contact us at info@headsin.co, call 9773497763, or use https://headsin.co/contact-us.\n"
        "- If asked about social media:\n"
        "  → Instagram: https://www.instagram.com/headsin.co | Facebook: https://www.facebook.com/people/HeadsInco/61574907748702/ | LinkedIn: https://www.linkedin.com/company/headsinco/\n"

        "Irrelevant Question Rule:\n"
        "- If the user's message is clearly unrelated to job search, hiring, or HeadsIn, politely say: 'I'm sorry, I can only assist with questions related to the HeadsIn platform.'\n"
        "- If this happens 3 times, reply once more and stop responding:\n"
        "  'Thanks for checking with us. For more information, visit:\n"
        "  Job Seekers: https://headsin.co/auth\n"
        "  Recruiters: https://company.headsin.co/auth'"
        " Any Other Questions: "
        "   Call : +91 9773497763"
        "   Email : info@headsin.co"
    )


    # Send to AI
    chat = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Use the following to answer the user query:\n\nRelevant info: {best_chunk}\n\nUser: {user_input}"}
        ]
    )
    answer = chat.choices[0].message.content.strip()

    # Handle irrelevant detection
    irrelevant_phrases = [
        "i'm sorry", "i can only assist", "only assist with", "not related to headsin"
    ]
    if any(phrase in answer.lower() for phrase in irrelevant_phrases):
        st.session_state.irrelevant_count += 1
    else:
        st.session_state.irrelevant_count = 0

    if st.session_state.irrelevant_count >= 3:
        answer = FINAL_BLOCK_MESSAGE

    # Store message history
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", answer))

# Show history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)
