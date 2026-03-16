import chromadb
import os
import pandas as pd
import pymupdf  #     pip install PyMuPDF
from groq import Groq  # hosted open-model provider
import threading
from dotenv import load_dotenv
from docx import Document # pip install python-docx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_assistant.utils.file_tracker import FileTracker
from dataclasses import dataclass

# TODO: Need to have a possiblity for hybrid search (semantic + keyword search), soemthing like following::
# 1. Hybrid (weighted fusion): add weighted hybrid (BM25 + 0.7/0.3 fusion).
# 2. Hybrid + Cross-Encoder rerank:: --> until you move to a cloud or GPU-enabled environment
#         *   Uses BM25 for fast keyword-based retrieval of relevant document chunks.
#         *   Uses semantic search (embeddings) for context-based retrieval.
#         *   Combines the top results from both methods and reranks them using a CrossEncoder model for improved relevance.
 
THRESHOLD = 0.3 # 0.3–0.5
N_RESULTS = 5
DB_PATH = 'results/chroma_db'

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
EMBED_MODEL = os.getenv('EMBED_MODEL', '')


@dataclass
class PageEntry:
    page_num: int | None
    text: str

@dataclass
class DocMetadata:
    # metadatas.append({"file": os.path.basename(file_path), "chunk": i, "page_num": page.page_num})
    file: str
    chunk: int
    page_num: int


class FileSearch:
    """Implementation for semantic search for a keyword in pdf/ word/ txt file"""
    _instance = None
    _lock = threading.Lock() 

    def __new__(cls):
        if cls._instance is None:
            # below lock is required in the case of multithreaded environment, Two threads could reach the if cls._instance is None: line at the same time before any of them assigns to _instance
            # we need to synchronize the critical section where _instance is being created.
            with cls._lock: 
                if cls._instance is None:     # <-- Check again inside lock
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return # prevent re-init (singleton) --> only one chroma connection open, avoids recreating collection.
        # self.loaded_files: dict = {}
        self.embedding_cache: dict = {}
        # 1. Start Chroma client (in-memory DB)
        # chroma_client = chromadb.Client()
        # persistant client:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        # 2. Create a collection if doesn't exists
        # "l2" → Euclidean distance, "ip" → Inner Product (works like similarity, higher = better), "cosine" → Cosine distance (lower = better)
        self.collection = chroma_client.get_or_create_collection(name="docs", metadata={"hnsw:space": "ip"})

        # 1. Load a pre-trained embedding model
        self.model = SentenceTransformer(EMBED_MODEL) # Small, fast, and good enough for search tasks

        # Text splitter for recursive chunking
        # This works on character counts.
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,      # adjust as needed (good balance)
        #     chunk_overlap=50,   # ensures semantic continuity
        #     separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        # )

        # Token based dynamic chunking:
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=600,     # measured in tokens (not characters)
            chunk_overlap=100
        )
        self._initialized = True # mark as initialized, this is done to prevent re initialisation of the same object (we already make it as singleton via means of overriding __new__)
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.file_tracker  = FileTracker()
    
    def load_file(self, file_path: str) -> None:
        """Load a file (txt, pdf, docx), extract text, and split into sentences."""
        if not os.path.isfile(file_path):
            print("File doesn't exists, please check path")
            return
        # check file loaded
        if self.file_tracker.check_file_loaded(file_path=file_path):
            # check if file is already indexed and it it is uptodate
            if self.file_tracker.is_indexed(file_path=file_path):
                print('file already exists')
                return
            else:
                # file outdated, need to remove from both vector db and json
                print(f"{file_path} has changed — re-indexing...")
                self._delete_from_db()
                
        if file_path.endswith('.txt'):
            pages = self._extract_text_from_text(file_path)
        elif file_path.endswith('.pdf'):
            pages = self._extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            pages = self._extract_text_from_word(file_path)
        else:
            print('Unsupported file type')
            return
        self._add_to_db(file_path, pages)

        # add file info to json/ sqlite db
        self.file_tracker.add_file(file_path=file_path)
    
    def _extract_text_from_pdf(self, file_path: str) -> list[PageEntry]:
        """Extract text from a PDF file"""
        doc = pymupdf.open(file_path)
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if len(text) > 10:  # ignore empty pages
                pages.append(PageEntry(page_num=page_num, text=text))
        return pages

    def _add_to_db(self, file_path: str, pages: list[PageEntry]):
        # Split text into smart chunks
        # Chunks per page -> embeds -> stores chunk + page metadata."""
        # pages -> page aware chunking (with overlap tokens + chunk_size as 600 + metadata (chunk no, pageno etc) Rebuild the context around a result (show surrounding text)) 
        # -> embeddings -> store in chroma(vector db) -> semantic search
        print(f'########## indexed pages:: {len(pages)}')
        for page in pages:

            chunks = self.text_splitter.split_text(page.text)
            print('########## ADDING chunks')
            ids = []
            metadatas = []
            # chunks:i is added as metadate here, as it can Trace back to the original document position (page, section, chunk).
            # Rebuild the context around a result (show surrounding text).
            for i in range(len(chunks)):

                ids.append(f"{os.path.basename(file_path)}_{page.page_num}_{i}")
                metadatas.append({"file": os.path.basename(file_path), "chunk": i, "page_num": page.page_num})
                # metadatas.append(DocMetadata(file = os.path.basename(file_path), chunk = i, page_num = page.page_num))

            # Step 2: Convert sentences into embeddings (vectors)
            # sentence_embeddings = self.model.encode(docs).tolist()  # Encodes all documents (or text chunks) at once in a single batch, used when the list docs is small 
            # Embed in batches (efficient)
            sentence_embeddings = self.model.encode(chunks, batch_size=16, show_progress_bar=True).tolist() # Encodes the input in smaller batches of 16 chunks at a time

            # Add to Chroma (with metadata)
            # requires one list of IDs and one list of metadatas aligned with each document
            # Thread safe db add
            with FileSearch._lock:
                self.collection.add(
                    documents=chunks,
                    embeddings=sentence_embeddings,
                    ids= ids,
                    metadatas=metadatas
                )
            # for tracking
            # self.loaded_files[file_path] = len(chunks)
            print(f"✅ Indexed {len(chunks)} chunks from {page.page_num}")

        return
        
    def _extract_text_from_word(self, file_path: str) -> None:
        """Extract text from word (.docx) file"""
        pages = []
        doc = Document(file_path)
        file_contents =  '\n'.join(para.text for para in doc.paragraphs if para.text.strip())
        return pages.append(PageEntry(page_num=None, text=file_contents))

    def _extract_text_from_text(self, file_path: str) -> None:
        """Extract text from a .txt file"""
        pages = []
        with open (file_path, 'r', encoding='utf-8', errors='replace') as file:
            file_contents = file.read()
        return pages.append(PageEntry(page_num=None, text=file_contents))

    def _get_embedding(self, search_key: str) -> str:
        """Return cached embedding if available."""
        search_key = search_key.lower()
        if search_key in self.embedding_cache:
            return self.embedding_cache[search_key]
        embedding = self.model.encode([search_key]).tolist()
        self.embedding_cache[search_key] = embedding
        return embedding
        
    def __keyword_Search(self, search_key:str) -> list:
        """Search for keyword in stored sentences."""
        # step4: query
        query_embedding = self._get_embedding(search_key)
        # 7. Search in Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS # return top 5
        )

        filtered_results = []
        # filtering response based on cosine similarity score
        for doc, score, meta in zip(results['documents'][0],results['distances'][0], results["metadatas"][0]):
            if score >= THRESHOLD:
                # print({"file": meta["file"], "score": score, "text": doc})
                # filtered_results.append(doc)
                filtered_results.append({"file": meta["file"], 'page': meta['page_num'], 'chunk': meta['chunk'], "score": score, "text": doc})
        return filtered_results
        
    # def _file_already_loaded(self, file_name: str) -> bool:
    #     # Return all documents whose metadata has a key file equal to <file_name>
    #     results = self.collection.get(where={'file': file_name})
    #     if len(results['ids']) > 0:
    #         return True
    #     return False

    def _delete_from_db(self, file_path:str) -> None:
        """ delete entries from vector db """
        with FileSearch._lock:
            self.collection.delete(where={"file": os.path.basename(file_path)})

    def keyword_Search(self, query: str):
        """Generate an answer from retrieved context using Ollama (RAG)"""
        # Step 1: Get top semantic matches
        search_results = self.__keyword_Search(query)

        if not search_results:
            return "No relevant context found in the indexed documents."

        # Step 2: Build the context from top chunks
        # context = '\n\n'.join(data['text'] for data in search_results)
        context = ''
        for data in search_results:
            context += f'File: {data['file']} | Page: {data['page']} \n{data['text']} \n'
        # # Step 3: Create a RAG-style prompt
        # prompt = f"""You are a helpful assistant.
        # Use ONLY the context below to answer the question accurately and concisely.
        # If the answer is not contained in the context, say: "The context does not provide enough information."
        
        # Context:
        # {context}

        # Question: {query}
        # Answer:
        # """

        # Step 4: Call llm model
        chat_completion = self.groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based only on provided context. If the answer is not explicitly stated, say ‘Not found in the text.’"},
            {'role': 'user', 'content': f'Context: \n {context}\n Question: {query} Answer with citations like: (File: X, Page: Y)'}], model=LLM_MODEL,)
        # print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
        
    def list_files(self):
        """List all loaded files"""
        return self.file_tracker.list_files()

if __name__ == '__main__':
    import time
    search = FileSearch()
    # search.load_file(file_path='/home/geethuthomas/python_samples/document_analyser/document_assistant/data/advs.pdf')
    search_key = 'how holmes solved the case'
    # search_key1 = 'ai'
    # search.keyword_Search(search_key)

    search.keyword_Search(search_key)
