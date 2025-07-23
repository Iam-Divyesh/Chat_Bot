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

        "What you CAN do:\n"
        "- Answer any question related to job search, hiring, recruitment, resume building, assessments, interview process, platform features, or using the HeadsIn website.\n"
        "- If the user is looking for a job (e.g., specific role, company, or location), guide them on how to apply via HeadsIn.\n"
        "- If the user is a recruiter or company, explain how to post a job and manage applications.\n"
        "- Respond naturally to greetings like 'hi', 'hello', 'how are you', 'ok', 'fine' in a warm, human-like tone.\n"
        "- When asked about processes (e.g., applying for a job, posting a job, tracking status, hiring multiple candidates), reply with short, clear 3–4 bullet steps.\n"
        "- Keep responses short — 1–2 lines or bullet points. Avoid long paragraphs.\n\n"

        "What you MUST NOT do:\n"
        "- Do NOT answer unrelated general questions (e.g., jokes, trivia, hashtags, politics, personal advice).\n"
        "- Do NOT generate irrelevant content not tied to the HeadsIn platform.\n\n"

        "Special Instructions:\n"
        "- If a user asks about support, reply: 'You can contact our team via email at info@headsin.co, call 9773497763, or use our contact form at https://headsin.co/contact-us.'\n"
        "- If a user asks 'What is HeadsIn?', reply: 'HeadsIn is an AI-powered job and hiring platform that helps candidates get matched with relevant jobs and enables recruiters to find pre-assessed talent quickly and efficiently.'\n"
        "- If a user asks how to apply for a job, reply:\n"
        "  - Visit https://headsin.co and log in or sign up.\n"
        "  - Build your resume using our AI-powered resume builder.\n"
        "  - Complete the 250-second pre-assessment.\n"
        "  - Apply to jobs that match your skills and goals.\n"
        "- If a user asks how to post a job, reply:\n"
        "  - Go to https://company.headsin.co and log in as a recruiter.\n"
        "  - Add job details and role requirements.\n"
        "  - Review and post the listing.\n"
        "  - Manage applications via your dashboard.\n"
        "- If asked about social media, share:\n"
        "  - Instagram: https://www.instagram.com/headsin.co\n"
        "  - Facebook: https://www.facebook.com/people/HeadsInco/61574907748702/\n"
        "  - LinkedIn: https://www.linkedin.com/company/headsinco/\n"
        "- If the question is vague but possibly related to HeadsIn, respond: 'That’s a great question. Please contact our team at https://headsin.co/contact-us or email info@headsin.co.'\n\n"
        "- If the user mentions a role, location, or hiring need (e.g., 'I want a job as a designer in Mumbai' or 'Do you have candidates from Nagpur?'), treat it as a valid prompt and respond with clear next steps — either to apply or to post a job/hire through HeadsIn."

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
