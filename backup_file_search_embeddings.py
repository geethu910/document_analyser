import chromadb
import os
import pandas as pd
import fitz  #     pip install PyMuPDF
import uuid
from docx import Document # pip install python-docx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class FileSearch:
    """Implementation for semantic search for a keyword in pdf/ word/ txt file"""

    def __init__(self):
        self.loaded_files = {}
        # 1. Start Chroma client (in-memory DB)
        # chroma_client = chromadb.Client()
        # persistant client:
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        # 2. Create a collection if doesn't exists
        self.collection = chroma_client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})

        # 1. Load a pre-trained embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # Small, fast, and good enough for search tasks

    def load_file(self, file_path: str) -> None:
        """Load a file (txt, pdf, docx), extract text, and split into sentences."""
        if not os.path.isfile(file_path):
            print("File doesn't exists, please check path")
            return
        if file_path.endswith('.txt'):
            self._extract_text_from_text(file_path)
        elif file_path.endswith('.pdf'):
            self._extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            self._extract_text_from_word(file_path)
        else:
            print('Unsupported file type')
    
    def _extract_text_from_pdf(self, file_path: str):
        """Extract text from a PDF file"""
        docs = self.__extract_paragraphs_pdf(file_path)
        self.loaded_files[file_path] = len(docs)
        ids = []
        metadatas = []
        # chunks:i is added as metadate here, as it can Trace back to the original document position (page, section, chunk).
        # Rebuild the context around a result (show surrounding text).
        for i, _ in enumerate(docs):
            ids.append(f"{os.path.basename(file_path)}_{i}")
            metadatas.append({"file": os.path.basename(file_path), "chunk": i})

        # Step 2: Convert sentences into embeddings (vectors)
        sentence_embeddings = self.model.encode(docs).tolist()
        
        # 5. Add sentences to Chroma with embeddings
        # requires one list of IDs and one list of metadatas aligned with each document.
        self.collection.add(documents=docs, embeddings=sentence_embeddings,
                       ids=ids, metadatas=metadatas)
    
    def _extract_text_from_word(self, file_path: str):
        """Extract text from word (.docx) file"""
        doc = Document(file_path)
        docs = [para.text for para in doc.paragraphs if para.text.strip()]
        self.loaded_files[file_path] = len(docs)
        sentence_embeddings = self.model.encode(docs).tolist()
        self.collection.add(
            documents=docs,
            embeddings=sentence_embeddings,
            ids=[f"{os.path.basename(file_path)}_{uuid.uuid4()}"  for i in range(len(docs))]
        )

    def _extract_text_from_text(self, file_path: str) -> None:
        """Extract text from a .txt file"""
        with open (file_path, 'r') as file:
            file_contents = file.read()
        
        docs = [p.strip() for p in file_contents.split("\n") if p.strip()]
        self.loaded_files[file_path] = len(docs)
        # Step 2: Convert sentences into embeddings (vectors)
        sentence_embeddings = self.model.encode(docs).tolist()

        # 5. Add sentences to Chroma with embeddings
        self.collection.add(documents=docs, embeddings=sentence_embeddings,
                       ids=[f"{os.path.basename(file_path)}_{uuid.uuid4()}" for i in range(len(docs))]) #Use unique IDs (e.g., include file name or hash).
    

    def keyword_Search(self, search_key:str) -> list:
        """Search for keyword in stored sentences."""
        # step4: query
        query_embedding = self.model.encode([search_key]).tolist()

        # 7. Search in Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5 # return top 5
        )
        for doc, dist in zip(results['documents'][0],results['distances'][0]):
            print(f"{dist:.3f} → {doc}")
        return results['documents'][0]
        
    def __extract_paragraphs_pdf(self, file_path):
        """Extracts paragraph from a pdf file"""
        doc = fitz.open(file_path)
        paragraphs = []
        for page in doc:
            blocks = page.get_text("blocks")  # returns list of text blocks
            for b in blocks:
                text = b[4].strip()
                if len(text.split()) > 3:  # filter out junk (like page numbers)
                    paragraphs.append(text)
        return paragraphs
