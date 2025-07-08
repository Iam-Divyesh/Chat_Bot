import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pdfplumber
import numpy as np
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Headsin Chatbot", page_icon="🤖")

# Load environment variables
load_dotenv()

# Azure OpenAI Setup
endpoint = "https://job-recruiting-bot.openai.azure.com/"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return clean_text(text)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# Cache embeddings to avoid recomputing on every run
@st.cache_resource
def prepare_embeddings():
    website_urls = [
        "https://headsin.co/",
        "https://headsin.co/code-of-conduct",
        "https://headsin.co/privacy-policy",
        "https://headsin.co/terms-and-conditions",
        "https://headsin.co/about-us",
        "https://headsin.co/contact-us",
        "https://headsin.co/build-resume-page",
        "https://headsin.co/candidate",
        "https://headsin.co/company"
    ]

    web_text = ""
    for url in website_urls:
        web_text += scrape_page(url) + "\n"

    with pdfplumber.open("./HeadsIn_Public_Chatbot_Report_2025.pdf") as pdf:
        text = ''
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    combined_text = text + "\n" + web_text
    chunks = [clean_text(chunk) for chunk in chunk_text(combined_text) if chunk.strip() != '']

    chunk_embeddings = []
    for idx, chunk in enumerate(chunks):
        if chunk.strip() == '':
            continue

        response = client.embeddings.create(
            model=embedding_deployment,
            input=[chunk]
        )

        embedding = response.data[0].embedding
        chunk_embeddings.append({"text": chunk, "embedding": embedding})

    return chunk_embeddings

# Load embeddings
chunk_embeddings = prepare_embeddings()


st.title("🤖 Headsin's Chatbot")
# Initialize chat history with welcome message
if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hello! How can I help you today? 😊")]

# User input field
user_input = st.chat_input("Ask me anything...")

if user_input:
    cleaned_user_input = clean_text(user_input)

    user_embedding_response = client.embeddings.create(
        model=embedding_deployment,
        input=[cleaned_user_input]
    )

    user_embedding = user_embedding_response.data[0].embedding

    best_chunk = None
    best_score = -1

    for item in chunk_embeddings:
        score = cosine_similarity(user_embedding, item["embedding"])
        if score > best_score:
            best_score = score
            best_chunk = item["text"]

    system_prompt = (
        "You are a helpful, professional chatbot dedicated only to answering questions about HeadsIn company. "
        "You are NOT allowed to answer general knowledge, social media, creative writing, or unrelated questions. "
        "You may only answer questions if they are clearly about HeadsIn's services, policies, platform, features, or website. "
        "If the user asks anything unrelated to HeadsIn (like general hashtags, trivia, or jokes), reply with:\n"
        "'I'm sorry, I can only assist with questions specifically related to HeadsIn company.'\n"
        "If the question is about HeadsIn but you don’t have enough information, respond:\n"
        "'That’s a great question. Please contact our team at https://headsin.co/contact-us or email contact@headsin.co for the most accurate information.'\n"
        "Always be polite, concise, and human-like in your answers. "
        "If asked about social media, you can share:\n"
        "- Instagram: https://www.instagram.com/headsin.co\n"
        "- Facebook: https://www.facebook.com/people/HeadsInco/61574907748702/\n"
        "- LinkedIn: https://www.linkedin.com/company/headsinco/"
    )



    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Use the following information to answer:\n{best_chunk}\n\nQuestion: {user_input}"}
        ]
    )

    assistant_response = response.choices[0].message.content

    # ✅ Save correctly as user and assistant
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", assistant_response))

# ✅ Display chat history correctly
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)
