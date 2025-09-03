



import re
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import logging
from dotenv import load_dotenv
from collections import deque
from serpapi import GoogleSearch  # 🆕 New import
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from chatbot_engine import ChatEngine


# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # 🆕 Added

INDEX_NAME = "chatbot-index"
DATA_FOLDER = "data"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- openai Setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Tokenizer ---
encoding = tiktoken.get_encoding("cl100k_base")

# --- Session Memory ---
user_sessions = {}

# --- Embedding ---
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",   # or "text-embedding-3-large"
        input=text
    )
    return response.data[0].embedding




# --- PDF Reading ---
def read_pdf(filepath):
    text = ""
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
            for _ in image_blocks:
                text += f"\n[Image on page {page_num+1}: Possibly a graph or chart. Provide context if relevant.]\n"
            text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
    return text

# --- Token-based Chunking ---
def split_text_by_tokens(text, max_tokens=400):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
    return chunks


def sanitize_id(text: str) -> str:
    # 1. Remove all non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]", "", text)
    # 2. Replace spaces with underscores (optional, but cleaner)
    text = text.replace(" ", "_")
    return text


# --- SerpAPI Search for Links --- 
def search_google(query):
    try:
        print(f"🔎 Searching Google for: {query}")
        
        # Ensure the API key is available
        if not SERPAPI_API_KEY:
            print("⚠️ No SERPAPI_API_KEY found")
            return ""
            
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "num": 3,  # Increased to 3 results
            "hl": "en",
            "gl": "us"  # Added region
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        links = []
        snippets = []
        
        # Get organic result links and snippets
        organic_results = results.get("organic_results", [])[:3]
        for result in organic_results:
            if result.get("link"):
                links.append(result["link"])
                if result.get("snippet"):
                    snippets.append(result["snippet"])

        # Get answer box content if available
        if "answer_box" in results:
            answer_box = results["answer_box"]
            if answer_box.get("answer"):
                snippets.insert(0, answer_box["answer"])
            elif answer_box.get("snippet"):
                snippets.insert(0, answer_box["snippet"])
            if answer_box.get("link"):
                links.insert(0, answer_box["link"])

        # Format the response
        response_parts = []
        
        # Add snippets if available
        if snippets:
            response_parts.append("\nKey information from web:")
            response_parts.extend(f"• {snippet}" for snippet in snippets)
            
        # Add links
        if links:
            response_parts.append("\nSources:")
            response_parts.extend(f"• {link}" for link in links)
            
        final_response = "\n".join(response_parts)
        print(f"✅ Found {len(links)} sources and {len(snippets)} snippets")
        return final_response if final_response else ""
            
    except Exception as e:
        print(f"❌ Google Search error: {str(e)}")
        logger.error(f"Google Search error: {e}")
        return ""



def clean_response(text):
    # Remove bold (**...**)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # Remove italic (*...*)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text.strip()
# --- Chat Logic ---
def chat_with_bot(query, memory_context="", chat_engine=None):
    # Check for greetings first
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if query.lower().strip() in greetings:
        return "Hello! How can I assist you today?"


    # Fetch context directly from DB
    document_context = ""
    if chat_engine:
        row = chat_engine.fetch_context(chat_engine.embed_text(query))
        if row and row[2].strip():   # only assign if real text exists
            document_context = row[2].strip()


#         prompt = f"""
# You are an AI assistant built by WLGA company.

# If a user asks about your role, purpose, or how you can help — 
# such as questions like "What can you do?" or "How can you assist me?" — 
# you should reply with:

# "I can assist you to know about WLGA company."

# Now, answer the user's query below:
# User: {query}
# """
#     else:
        prompt = f"""
Answer the user's question in a clear, helpful tone using the information below. If not found, use your own knowledge.
 You are a knowledgeable and helpful AI assistant.

 You can use the information below to answer. If the exact answer is not found, respond using your own knowledge — but never mention whether the information came from a document or memory. Just provide the best possible answer confidently and naturally.

 Follow these formatting rules:
 - Keep paragraphs short and break them for readability when needed.
 - If the user asks about your knowledge, capabilities, or how many documents/sources you contain, respond naturally. Do not use phrases like \"I do not\", \"I cannot\", \"I don’t have access\", or mention limitations.
 - You have to understand the pattern of language and then respond.
 - Just give a general, helpful, confident answer — assuming you are capable of answering from a wide range of relevant knowledge.
 - Use headings or sub-headings only when they help clarify.
 - Restrictly don't use ( * , ** ) charaters just bold that sentences.
 - If bullet points are required, place each on a new line (no special characters or numbering).
 - Never include phrases like \"the document says\", \"based on the context\", \"I do not know\", \"as an AI\", or any reference to documents, memory, or limitations.

Chat history:
{memory_context}

Reference:
{document_context}

User's question:
{query}

Answer:
"""
      
    # Get answer primarily from documents
        print(f"📝 Query: {query}")
        print(f"📚 Document context available: {bool(document_context)}")
        
        if not document_context:
            return "I apologize, but I need to find relevant information in our LPG documents to answer your question. Could you please rephrase your question to focus on LPG-related topics covered in our documents?"

        # First, get a detailed answer from documents
        doc_prompt = f"""
You are an expert in LPG (Liquid Petroleum Gas) industry. Generate a comprehensive answer using PRIMARILY the document information provided.
Focus 80% on the document content and only briefly supplement with general knowledge if necessary.

Document Reference:
{document_context}

User's question:
{query}

Instructions:
1. Focus mainly on information from the provided document
2. Be specific and detailed with document information
3. Keep any additional/general information brief and minimal
4. Format the answer in clear paragraphs
5. Do not mention sources or references in the answer
"""
        doc_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are an LPG industry expert. Focus primarily on document information."
            },
            {
                "role": "user",
                "content": doc_prompt
            }]
        )
        
        doc_answer = doc_response.choices[0].message.content.strip()
        doc_answer = clean_response(doc_answer)
        print(f"📘 Generated document-based answer: {len(doc_answer)} chars")

        # Get minimal Google results to supplement
        try:
            print("🔍 Searching Google for supplementary info...")
            google_results = search_google(f"LPG {query}")
            print(f"🌐 Google results found: {bool(google_results)}")
            
            if google_results:
                # Get a brief summary of Google results
                summary_prompt = f"""
Summarize the most important additional information from these search results in 2-3 brief points.
Current answer from documents:
{doc_answer}

Search results:
{google_results}

Give only new information that's not already covered in the document answer.
Keep it very brief (max 2-3 lines).
"""
                summary_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                google_summary = summary_response.choices[0].message.content.strip()
                
                # Combine answers with clear separation
                final_answer = f"{doc_answer}\n\nAdditional relevant information:\n{google_summary}\n\nSources for further reading:\n{google_results}"
            else:
                final_answer = doc_answer
                
            print(f"✅ Returning answer: {len(final_answer)} chars")
            return final_answer
            
        except Exception as e:
            print(f"❌ Error during Google search: {str(e)}")
            return doc_answer  # Return the document-based answer if Google search fails
# --- Intent Detection ---
def detect_intent(user_query):
    clear_phrases = [
        "clear", "reset", "delete", "start over", "new chat", "forget everything",
        "clear chat", "clear conversation", "reset history", "delete all","refresh chat","delete history","refresh"
    ]
    user_query = user_query.lower()
    return "clear_chat" if any(phrase in user_query for phrase in clear_phrases) else "chat"



#  --- Flask Setup ---
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    query = data.get("query", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if detect_intent(query) == "clear_chat":
        user_sessions.pop(session_id, None)
        return jsonify({"response": "", "session_id": str(uuid.uuid4())})

    if session_id not in user_sessions:
        user_sessions[session_id] = deque(maxlen=5)

    user_sessions[session_id].append({"role": "user", "text": query})
    memory_context = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in user_sessions[session_id]])

    answer = chat_with_bot(query, memory_context, chat_engine)
    user_sessions[session_id].append({"role": "bot", "text": answer})

    return jsonify({"response": answer, "session_id": session_id})



load_dotenv()  # Loads values from .env into environment variables

if __name__ == "__main__":
    chat_engine = ChatEngine(
        db_host=os.getenv("DB_HOST"),
        db_name=os.getenv("DB_NAME"),
        db_user=os.getenv("DB_USER"),
        db_password=os.getenv("DB_PASSWORD"),
        db_port=os.getenv("DB_PORT"),
        table_name=os.getenv("TABLE_NAME")
    )

        
    print("✅ Chat engine initialized!")
    print("🚀 Starting the server on port 5000...")
    app.run(debug=False, host="0.0.0.0", port=5000)










