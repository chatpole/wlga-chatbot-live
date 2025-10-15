"""
WLGA Chatbot Backend - Clean Implementation
Simple flow: Check PostgreSQL DB first, if not found then OpenAI fallback
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import uuid
import psycopg2
from psycopg2 import pool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session storage (simple in-memory storage)
user_sessions = {}


class PostgreSQLDatabase:
    """PostgreSQL with pgvector integration for document search"""
    
    def __init__(self):
        self.openai_client = openai_client
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_name = os.getenv("DB_NAME", "mydb")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "secret")
        self.db_port = int(os.getenv("DB_PORT", 5432))
        self.table_name = os.getenv("TABLE_NAME", "documents")
        self.pool = None
        
        # Initialize database connection
        self._init_database()
        
    def _init_database(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=self.db_host,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port,
                sslmode="require"
            )
            logger.info(f"PostgreSQL connected successfully to {self.db_host}:{self.db_port}/{self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.pool = None
        
    def create_embedding(self, text):
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    def search_documents(self, query, top_k=3):
        """
        Search for relevant documents in PostgreSQL with pgvector
        Returns: List of relevant document chunks with metadata
        """
        try:
            if not self.pool:
                logger.info("PostgreSQL not available - skipping database search")
                return []
            
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return []
            
            logger.info(f"Searching PostgreSQL for: {query}")
            
            # Get connection from pool
            conn = self.pool.getconn()
            try:
                cur = conn.cursor()
                
                # Search for similar documents using pgvector
                cur.execute(f"""
                    SELECT id, filename, content, embedding <=> %s::vector AS distance
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_embedding, query_embedding, top_k))
                
                rows = cur.fetchall()
                cur.close()
                
                # Process results
                documents = []
                for row in rows:
                    doc_id, filename, content, distance = row
                    # Only include high-confidence matches (low distance = high similarity)
                    if distance < 0.8:  # Adjust threshold as needed
                        documents.append({
                            "content": content,
                            "metadata": {
                                "id": doc_id,
                                "filename": filename,
                                "distance": distance
                            },
                            "score": 1 - distance  # Convert distance to similarity score
                        })
                
                logger.info(f"Found {len(documents)} relevant documents in PostgreSQL")
                return documents
                
            finally:
                self.pool.putconn(conn)
            
        except Exception as e:
            logger.error(f"Error searching PostgreSQL: {e}")
            return []


class ChatbotEngine:
    """Main chatbot engine with clean logic"""
    
    def __init__(self):
        self.database = PostgreSQLDatabase()
        self.openai = openai_client
        
    def process_query(self, query, session_id=None):
        """
        Main processing logic:
        1. Check PostgreSQL DB first
        2. If not found or AI says no relevant info, use OpenAI fallback
        3. Return response with source information
        """
        try:
            # Step 1: Search PostgreSQL database
            logger.info(f"Processing query: {query}")
            db_results = self.database.search_documents(query)
            
            if db_results:
                # Found in database - try to use PostgreSQL results
                db_response = self._handle_database_response(query, db_results)
                
                # Check if AI indicated no relevant information
                if self._is_no_info_response(db_response["response"]):
                    logger.info("AI indicated no relevant info in database, using OpenAI fallback")
                    return self._handle_openai_fallback(query)
                else:
                    return db_response
            else:
                # Not found in database - use OpenAI fallback
                logger.info("No documents found in database, using OpenAI fallback")
                return self._handle_openai_fallback(query)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "source": "error",
                "details": f"Processing error: {str(e)}"
            }
    
    def _is_no_info_response(self, response):
        """Check if the AI response indicates no relevant information found"""
        no_info_indicators = [
            "i'm sorry", "i apologize", "information provided does not include",
            "no data", "not available", "cannot find", "don't have",
            "not in the documents", "not found in", "unable to find",
            "does not contain", "no information", "not include"
        ]
        return any(indicator in response.lower() for indicator in no_info_indicators)
    
    def _handle_database_response(self, query, db_results):
        """Handle response when data is found in PostgreSQL database"""
        try:
            # Combine all relevant document chunks
            context_parts = []
            sources = []
            
            for result in db_results:
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                filename = metadata.get("filename", "Unknown document")
                
                if content:
                    context_parts.append(content)
                    sources.append(filename)
            
            if not context_parts:
                # Fallback to OpenAI if no content found
                return self._handle_openai_fallback(query)
            
            # Combine context
            combined_context = "\n\n".join(context_parts)
            
            # Generate response using OpenAI
            prompt = f"""
You are a helpful LPG (Liquid Petroleum Gas) industry expert. Use the following information from our documents to answer the user's question.

Document Information:
{combined_context}

User Question: {query}

Instructions:
- Answer based primarily on the provided document information
- Be clear, helpful, and accurate
- If the documents don't contain enough information, say so
- Don't mention that you're using documents - just provide the answer naturally

Answer:
"""
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable LPG industry expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "response": answer,
                "source": "database",
                "details": f"Answer based on {len(db_results)} document(s) from database",
                "sources": list(set(sources))  # Remove duplicates
            }
            
        except Exception as e:
            logger.error(f"Error handling database response: {e}")
            # Fallback to OpenAI
            return self._handle_openai_fallback(query)
    
    def _handle_openai_fallback(self, query):
        """Handle response using OpenAI when database has no relevant information"""
        try:
            logger.info(f"Using OpenAI fallback for query: {query}")
            
            # Generate LPG-focused response using OpenAI
            prompt = f"""
You are a knowledgeable LPG (Liquid Petroleum Gas) industry expert. The user has asked a question that may not be directly covered in our specific documents, but you should provide helpful information related to LPG industry.

User Question: {query}

Instructions:
- Provide a helpful, informative answer about LPG industry topics
- Focus on LPG-related information and industry insights
- Be educational and informative about LPG applications, safety, benefits, etc.
- If the question is not directly LPG-related, try to connect it to LPG industry context
- Be confident and knowledgeable in your response
- Don't apologize for not having specific documents - provide valuable LPG industry information

Answer:
"""
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a confident and knowledgeable LPG industry expert. Always provide helpful, informative answers about LPG topics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "response": answer,
                "source": "openai_fallback",
                "details": "Answer generated using OpenAI general knowledge (database had no relevant information)"
            }
            
        except Exception as e:
            logger.error(f"Error in OpenAI fallback: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try again.",
                "source": "error",
                "details": f"OpenAI fallback error: {str(e)}"
            }


# Initialize chatbot engine
chatbot = ChatbotEngine()


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Main chat endpoint"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Handle special commands
        if query.lower() in ["clear", "reset", "clear chat", "new chat"]:
            session_id = data.get("session_id")
            if session_id and session_id in user_sessions:
                del user_sessions[session_id]
            return jsonify({
                "response": "Chat cleared successfully.",
                "source": "system",
                "details": "Chat session cleared",
                "session_id": str(uuid.uuid4())
            })
        
        # Handle greetings
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if query.lower().strip() in greetings:
            return jsonify({
                "response": "Hello! I'm your LPG industry assistant. How can I help you today?",
                "source": "greeting",
                "details": "Direct greeting response",
                "session_id": data.get("session_id", str(uuid.uuid4()))
            })
        
        # Get or create session
        session_id = data.get("session_id", str(uuid.uuid4()))
        if session_id not in user_sessions:
            user_sessions[session_id] = []
        
        # Store user query in session
        user_sessions[session_id].append({"role": "user", "content": query})
        
        # Process query
        result = chatbot.process_query(query, session_id)
        
        # Store bot response in session
        user_sessions[session_id].append({"role": "assistant", "content": result["response"]})
        
        # Add session ID to result
        result["session_id"] = session_id
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "WLGA Chatbot API is running",
        "database": "PostgreSQL with pgvector",
        "fallback": "OpenAI general knowledge"
    })


@app.route("/", methods=["GET"])
def home():
    """Basic root route"""
    return jsonify({
        "message": "Backend service running successfully!",
        "status": "OK"
    }), 200


"""
if __name__ == "__main__":
    logger.info("Starting WLGA Chatbot Backend...")
    logger.info("Flow: PostgreSQL DB check → OpenAI fallback")
    app.run(debug=False, host="0.0.0.0", port=8000)
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chatbot_backend:app", host="0.0.0.0", port=port)



