# import os
# import psycopg2
# from PyPDF2 import PdfReader
# from openai import OpenAI
# from tabulate import tabulate

# # --- OpenAI client (direct API key) ---
# client = OpenAI(api_key="")  # 🔑 apni asli key yahan daalo

# # --- Database connection (RDS details) ---
# conn = psycopg2.connect(
#     host="database-1.cip8y4ck6n6j.us-east-1.rds.amazonaws.com",
#     port=5432,
#     dbname="postgres",
#     user="postgres",
#     password="Bhuwan12345"
# )
# cur = conn.cursor()

# # --- Paths ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_FOLDER = os.path.join(BASE_DIR, "../data")

# # Prepare master table for all files
# master_table_data = []

# # --- Loop through all PDF files in data folder ---
# pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

# for file_num, pdf_file in enumerate(pdf_files, start=1):
#     file_path = os.path.join(DATA_FOLDER, pdf_file)
#     filename_only = os.path.splitext(pdf_file)[0]

#     # Read PDF
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"

#     # Chunk text (1000 characters per chunk)
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

#     # Process each chunk
# import os
# import psycopg2
# from PyPDF2 import PdfReader
# from openai import OpenAI
# from tabulate import tabulate

# # --- OpenAI client (direct API key) ---
# client = OpenAI(api_key="s")  # 🔑 apni asli key yahan daalo

# # --- Database connection (RDS details) ---
# conn = psycopg2.connect(
#     host="database-1.cip8y4ck6n6j.us-east-1.rds.amazonaws.com",
#     port=5432,
#     dbname="postgres",
#     user="postgres",
#     password="Bhuwan12345"
# )
# cur = conn.cursor()

# # --- Paths ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_FOLDER = os.path.join(BASE_DIR, "../data")

# # Prepare master table for all files
# master_table_data = []

# # --- Loop through all PDF files in data folder ---
# pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

# for file_num, pdf_file in enumerate(pdf_files, start=1):
#     file_path = os.path.join(DATA_FOLDER, pdf_file)
#     filename_only = os.path.splitext(pdf_file)[0]

#     # Read PDF
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"

#     # Chunk text (1000 characters per chunk)
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

#     # Process each chunk
#     for chunk in chunks:
#         if chunk.strip():
#             embedding = client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=chunk
#             ).data[0].embedding

#             # Insert into DB as VECTOR
#             cur.execute(
#                 "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s::vector)",
#                 (filename_only, chunk, embedding)
#             )

#             # Append preview data
#             master_table_data.append([
#                 file_num,
#                 filename_only,
#                 chunk[:80].replace("\n", " ") + "...",
#                 str(embedding[:5]) + "..."
#             ])

# # --- Finalize DB ---
# conn.commit()
# cur.close()
# conn.close()

# # --- Print Summary ---
# print("\n✅ All PDF embeddings stored successfully into pgvector!\n")
# print(tabulate(
#     master_table_data,
#     headers=["File No.", "Filename", "Content (preview)", "Embedding (preview)"],
#     tablefmt="fancy_grid"
# ))


# # --- Finalize DB ---
# conn.commit()
# cur.close()
# conn.close()

# # --- Print Summary ---
# print("\n✅ All PDF embeddings stored successfully into pgvector!\n")
# print(tabulate(
#     master_table_data,
#     headers=["File No.", "Filename", "Content (preview)", "Embedding (preview)"],
#     tablefmt="fancy_grid"
# ))




# # import os
# # import psycopg2
# # from PyPDF2 import PdfReader
# # from openai import OpenAI
# # import json
# # from tabulate import tabulate

# # # Initialize OpenAI client
# # client = OpenAI(api_key="")  # Replace with your key

# # # Database connection
# # conn = psycopg2.connect(
# #     host="database-1.cip8y4ck6n6j.us-east-1.rds.amazonaws.com",
# #     port=5432,
# #     dbname="postgres",
# #     user="postgres",
# #     password="Bhuwan12345"
# # )
# # cur = conn.cursor()

# # # Paths
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # DATA_FOLDER = os.path.join(BASE_DIR, "../data")

# # # Prepare master table for all files
# # master_table_data = []

# # # Loop through all PDF files in data folder
# # pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

# # for file_num, pdf_file in enumerate(pdf_files, start=1):
# #     file_path = os.path.join(DATA_FOLDER, pdf_file)
# #     filename_only = os.path.splitext(pdf_file)[0]

# #     # Read PDF
# #     reader = PdfReader(file_path)
# #     text = ""
# #     for page in reader.pages:
# #         text += page.extract_text() + "\n"

# #     # Chunk text
# #     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

# #     # Process each chunk
# #     for chunk in chunks:
# #         if chunk.strip():
# #             embedding = client.embeddings.create(
# #                 model="text-embedding-3-small",
# #                 input=chunk
# #             ).data[0].embedding

# #             # Insert into DB
# #             cur.execute(
# #                 "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s)",
# #                 (filename_only, chunk, json.dumps(embedding))
# #             )

# #             # Append one row to master table
# #             master_table_data.append([
# #                 file_num,
# #                 filename_only,
# #                 chunk[:80].replace("\n", " ") + "...",
# #                 str(embedding[:5]) + "..."
# #             ])

# # # Finalize DB
# # conn.commit()
# # cur.close()
# # conn.close()

# # # Print once at the end
# # print("\n✅ All PDF embeddings stored successfully!\n")
# # print(tabulate(
# #     master_table_data,
# #     headers=["File No.", "Filename", "Content (preview)", "Embedding (preview)"],
# #     tablefmt="fancy_grid"
# # ))