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

    with pdfplumber.open("./HeadsIn_Public_Chatbot_Report_2025.pdf") as pdf:
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
    "If you're seeking a job, visit: https://headsin.co/auth  \n"
    "If you're looking to hire candidates, go to: https://company.headsin.co/auth"
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
if user_input:
    cleaned_input = clean_text(user_input)

    # Get embedding for user input
    user_emb = client.embeddings.create(model=embedding_deployment, input=[cleaned_input])
    user_vec = user_emb.data[0].embedding

    # Find best matching chunk
    best_chunk = max(chunk_embeddings, key=lambda c: cosine_similarity(user_vec, c["embedding"]))["text"]

    # System prompt
    system_prompt = (
        '''You are a concise and professional assistant that only answers questions directly related to HeadsIn company.

        Rules:
        - Only answer if the question is about HeadsIn's services, platform, website, features, policies, or company info.
        - Do NOT answer general questions, creative writing, hashtags, jokes, random descriptions, or social media topics unless explicitly about HeadsIn.
        - Do NOT respond if the input is vague, descriptive, or not a clear question about HeadsIn.
        - Keep every answer in 1–2 sentences — short and succinct, helpful, and human-like.
        - If the user asks **how to use** something on HeadsIn (e.g., how to apply for a job, how to post a job), provide a short, structured, step-by-step answer in 3–4 bullet points.

        Unrelated prompt response (only for first 1–2 times):
        "I'm sorry, I can only assist with questions specifically related to HeadsIn company."

        If unsure about a valid HeadsIn query:
        "That’s a great question. Please contact our team at https://headsin.co/contact-us or email contact@headsin.co."

        If asked about social links, reply:
        - Instagram: https://www.instagram.com/headsin.co
        - Facebook: https://www.facebook.com/people/HeadsInco/61574907748702/
        - LinkedIn: https://www.linkedin.com/company/headsinco/

        If 3 or more irrelevant questions are asked, show this once and stop:
        Thanks for checking with us. Feel free to visit our website.

        If you're seeking a job, visit: https://headsin.co/auth  
        If you're looking to hire candidates, go to: https://company.headsin.co/auth'''
    )

    # Chat Completion
    chat = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Use the following to answer:\n{best_chunk}\n\nQuestion: {user_input}"}
        ]
    )
    answer = chat.choices[0].message.content

    # Check if irrelevant
    if "only assist with questions specifically related to HeadsIn" in answer:
        st.session_state.irrelevant_count += 1
    else:
        st.session_state.irrelevant_count = 0

    # Store message history
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", answer))

# Show history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)
