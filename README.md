# Basic RAG (Retrieval-Augmented Generation) Application with LangChain & Gemini/OpenAI

This guide summarizes a minimal example of a Retrieval-Augmented Generation (RAG) pipeline, using [LangChain v1.x](https://github.com/langchain-ai/langchain) and Gemini (GoogleGenerativeAI) or OpenAI embeddings for semantic search.

## Prerequisites

- Python 3.9+
- Install dependencies listed in your `requirements.txt`:
    - `langchain-core`, `langchain-google-genai`, `langchain-openai`, `langchain-community`, `faiss-cpu`, `chromadb`
    - Make sure your keys are set in `.env`:
        - `OPENAI_API_KEY` for OpenAI
        - `GOOGLE_API_KEY` for Gemini

## Workflow Steps

1. **Document Loading:** Use PyPDFLoader or .txt loader to ingest source documents.
2. **Text Splitting:** Split documents into manageable chunks.
3. **Embedding Creation:** Map each chunk into a vector using Gemini or OpenAI embedders.
4. **Vector Store Indexing:** Store embeddings in a vector database (FAISS/ChromaDB).
5. **Query Pipeline:** Receive user query, embed it, do semantic search, retrieve top matches, and generate a response using an LLM (e.g., Gemini or OpenAI).

---

## Example Code

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# 1. Load keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Load & split documents
loader = PyPDFLoader("attention.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
doc_splits = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=GEMINI_API_KEY)
# For OpenAI: from langchain_openai import OpenAIEmbeddings; embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Store embeddings in FAISS
vectorstore = FAISS.from_documents(doc_splits, embeddings)

# 5. Build Retrieval & LLM chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0)
prompt = ChatPromptTemplate.from_template("Given context: {context}\n\nAnswer the following: {question}")

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Use the RAG chain
query = "What is the Transformer architecture and its benefits?"
answer = rag_chain.invoke(query)
print(answer)
```

---

## What Happens?

1. **Document chunks** are indexed into a vector DB.
2. **Query** is embedded, nearest chunks are retrieved.
3. **Prompt** is formatted with context + question.
4. **LLM** generates an answer using both the retrieved context and its internal knowledge.

---

## Key Points

- For **embeddings**, use Gemini or OpenAI (or others, e.g., Sentence Transformers).
- For **vector store**, FAISS or ChromaDB are common.
- The **retriever** does semantic search, not just keyword match.
- The **LLM** can be Gemini or OpenAI, interacts via LangChain interface.

---

## Troubleshooting

- API connection errors might require valid SSL certificates or API key fixes.
- Library versions must be compatible (LangChain v1.x, latest embedders).
- Always supply mapping (`{"question": query}`) as input to the chain.

---

## References

- [LangChain RAG Docs](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain Google Generative AI Docs](https://python.langchain.com/docs/integrations/google_genai/)
- [LangChain Embeddings](https://python.langchain.com/docs/modules/data_connection/document_transformers/embeddings/)
