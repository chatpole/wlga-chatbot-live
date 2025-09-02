import os
import psycopg2
from psycopg2 import pool
from openai import OpenAI
from flask import jsonify

class ChatEngine:
    def __init__(self, 
                 db_host=None, db_name=None, db_user=None, db_password=None, db_port=None, 
                 table_name="documents"):

        # Load from env if not passed
        self.db_host = db_host or os.getenv("DB_HOST", "localhost")
        self.db_name = db_name or os.getenv("DB_NAME", "mydb")
        self.db_user = db_user or os.getenv("DB_USER", "postgres")
        self.db_password = db_password or os.getenv("DB_PASSWORD", "secret")
        self.db_port = db_port or int(os.getenv("DB_PORT", 5432))
        self.table_name = table_name

        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Connection pool
        self.pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=self.db_host,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            port=self.db_port,
            sslmode="require"
        )

    def embed_text(self, text):
        """Generate a 1536-dim embedding for input text using OpenAI."""
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return resp.data[0].embedding

    def fetch_context(self, query_vec):
        """Retrieve top 1 most similar row from pgvector table."""
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT id, filename, content, embedding <=> %s::vector AS distance
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
            """, (query_vec, query_vec))
            row = cur.fetchone()
            cur.close()
            return row
        finally:
            self.pool.putconn(conn)


    def respond(self, user_query):
        """Main entrypoint: retrieve context + generate GPT-4o response."""
        try:
            # Step 1: Embed user query
            query_vec = self.embed_text(user_query)

            # Step 2: Fetch nearest neighbor
            row = self.fetch_context(query_vec)
            if not row:
                return jsonify({"response": "I couldn't find any relevant context in the database."})

            context_text = row[2]  # content column
            source_name = row[1]   # filename

            # Step 3: Ask GPT-4o with RAG prompt
            rag_prompt = f"""
            You are a helpful assistant. Use the following context from documents to answer the question.

            Context:
            {context_text}

            Question:
            {user_query}

            Answer:
            """

            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant that answers using the given context."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=300
            )

            answer = completion.choices[0].message.content
            return jsonify({
                "response": f"Answer: {answer}\n\n(Source: {source_name})"
            })

        except Exception as e:
            return jsonify({
                "error": f"An error occurred: {str(e)}"
            }), 500

    def close(self):
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()


# import os
# import psycopg2
# import openai
# import numpy as np
# from flask import jsonify

# class ChatEngine:
#     def __init__(self, db_host, db_name, db_user, db_password, db_port, table_name="documents"):
#         # OpenAI API key
#         openai.api_key = os.getenv("OPENAI_API_KEY", "")

#         # Postgres connection
#         self.conn = psycopg2.connect(
#             host=db_host,
#             dbname=db_name,
#             user=db_user,
#             password=db_password,
#             port=db_port,
#             sslmode="prefer"
#         )
#         self.table_name = table_name

#     def embed_text(self, text):
#         """Generate a 1536-dim embedding for input text using OpenAI."""
#         resp = openai.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return resp.data[0].embedding

#     def fetch_context(self, query_vec):
#         """Retrieve top 1 most similar row from pgvector table."""
#         cur = self.conn.cursor()
#         cur.execute(f"""
#             SELECT id, filename, content, embedding <=> %s AS distance
#             FROM {self.table_name}
#             ORDER BY embedding <=> %s
#             LIMIT 1;
#         """, (query_vec, query_vec))
#         row = cur.fetchone()
#         cur.close()
#         return row  # (id, filename, content, distance)

#     def respond(self, user_query):
#         """Main entrypoint: retrieve context + generate GPT-4o response."""
#         try:
#             # Step 1: Embed user query
#             query_vec = self.embed_text(user_query)

#             # Step 2: Fetch nearest neighbor
#             row = self.fetch_context(query_vec)
#             if not row:
#                 return jsonify({"response": "I couldn't find any relevant context in the database."})

#             context_text = row[2]  # content column
#             source_name = row[1]   # filename

#             # Step 3: Ask GPT-4o with RAG prompt
#             rag_prompt = f"""
#             You are a helpful assistant. Use the following context from documents to answer the question.

#             Context:
#             {context_text}

#             Question:
#             {user_query}

#             Answer:
#             """

#             completion = openai.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": "You are a knowledgeable assistant that answers using the given context."},
#                     {"role": "user", "content": rag_prompt}
#                 ],
#                 max_tokens=300
#             )

#             answer = completion.choices[0].message["content"]
#             return jsonify({
#                 "response": f"Answer: {answer}\n\n(Source: {source_name})"
#             })

#         except Exception as e:
#             return jsonify({
#                 "error": f"An error occurred: {str(e)}"
#             }), 500

#     def close(self):
#         """Close the database connection."""
#         if self.conn:
#             self.conn.close()
