from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-chatbot'

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = vector_store.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
    # metadatas=[d.metadata for d in text_chunks],
    # ids=[str(uuid4()) for _ in range(len(text_chunks))]
    )