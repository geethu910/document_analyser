import asyncio
import chromadb
import os
import torch
import pandas as pd
import pymupdf  #     pip install PyMuPDF
from groq import Groq, AsyncGroq # hosted open-model provider
import threading
from dotenv import load_dotenv
from docx import Document # pip install python-docx
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_assistant import logger
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import time

# TODO: Need to have a possiblity for hybrid search (semantic + keyword search), soemthing like following::
# 1. Hybrid (weighted fusion): add weighted hybrid (BM25 + 0.7/0.3 fusion).
# 2. Hybrid + Cross-Encoder rerank:: --> until you move to a cloud or GPU-enabled environment
#         *   Uses BM25 for fast keyword-based retrieval of relevant document chunks.
#         *   Uses semantic search (embeddings) for context-based retrieval.
#         *   Combines the top results from both methods and reranks them using a CrossEncoder model for improved relevance.
 
THRESHOLD = 0.3 # 0.3–0.5
N_RESULTS = 5
CE_THRESHOLD = 0.1

load_dotenv()  # # only does something if .env exists (local dev)
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
EMBED_MODEL = os.getenv('EMBED_MODEL', '')
CROSS_ENCODER_MODEL = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

@dataclass
class PageEntry:
    page_num: int | None
    text: str

@dataclass
class DocMetadata:
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
            # need to synchronize the critical section where _instance is being created.
            with cls._lock: 
                if cls._instance is None:     # <-- Check again inside lock
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return # prevent re-init (singleton) --> only one chroma connection open, avoids recreating collection.
        # self.loaded_files: dict = {}
        self.embedding_cache: dict = {}
        # 1. Start Chroma client
        # persistant client:
        chroma_client = chromadb.HttpClient(host=os.getenv('CHROMA_HOST', 'localhost'), port=int(os.getenv('CHROMA_PORT', 8000)))
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

        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self._async_client = AsyncGroq(api_key=GROQ_API_KEY)
        self.bm25_indexes: dict = {}   # { user_id: BM25Okapi }
        self.bm25_chunks: dict = {}    # { user_id: [chunk texts] }
        self.bm25_ids: dict = {}       # { user_id: [chunk ids] }
        # self.reranker = CrossEncoder(CROSS_ENCODER_MODEL, activation_fn=torch.nn.Sigmoid()) # chunks are 600 token, 110 token got truncated
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL) 
        self._initialized = True # mark as initialized, this is done to prevent re initialisation of the same object (we already make it as singleton via means of overriding __new__)

    def load_file(self, file_path: str, user_id: str, file_hash: str, original_file_name: str) -> int:
        """Load a file (txt, pdf, docx), extract text, and split into sentences."""
        # Step 1: Check if already processed (by hash) --> checking if it is indexed by another users.
        start = time.perf_counter()
        logger.info("Started processing %s for user %s", original_file_name, user_id)
        if self._hash_exists_in_db(file_hash):
            logger.debug("Found existing vectors (hash: %s...)", file_hash[:8])
            return self._reuse_embeddings(file_hash, user_id, file_path, original_file_name)
        # Step 2: Process new file separately 
        # TODO: include _delete_db functionality incase the file is outdated
        if file_path.endswith('.txt'):
            pages = self._extract_text_from_text(file_path)
        elif file_path.endswith('.pdf'):
            pages = self._extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            pages = self._extract_text_from_word(file_path)
        else:
            logger.error('Unsupported file type')
            return 0
        chunk_count = self._add_to_db(file_path, pages, user_id, file_hash, original_file_name)
        elapsed = time.perf_counter() - start
        logger.info("Finished processing %s for user %s: chunks=%d time=%.3fs", original_file_name, user_id, chunk_count, elapsed)
        return chunk_count


    def _hash_exists_in_db(self, file_hash: str) -> bool:
        """Check if file with this hash already exists"""
        results = self.collection.get(where={"file_hash": file_hash})
        return len(results['ids']) > 0

    def _reuse_embeddings(self, file_hash: str, user_id: str, file_path: str, original_file_name: str) -> int:
        """Reuse existing embeddings for a new user (DEDUPLICATION)"""
        # Get all chunks for the given file hash
        start = time.perf_counter()
        results = self.collection.get(where={"file_hash": file_hash})
        if not len(results['ids']) > 0:
            return 0
        logger.debug(f"Reusing {len(results['ids'])} chunks for user {user_id}")
        new_ids = []
        new_metadatas = []
        user_id_str = str(user_id)

        for metadata in results['metadatas']:
            new_id = f"{user_id_str}_{original_file_name}_{metadata['page_num']}_{metadata['chunk']}"
            new_ids.append(new_id)

            # update metadata
            new_metadata = {
                "user_id": user_id_str,
                "file": original_file_name,
                "file_hash": file_hash,
                "chunk": metadata['chunk'],
                "page_num": metadata['page_num'],
                "shared_from": metadata.get("user_id", "original")
            }
            new_metadatas.append(new_metadata)
        # ADD TO DB ONCE (all chunks at once)
        with FileSearch._lock:
            self.collection.add(
                ids=new_ids,
                embeddings=results['embeddings'],
                documents=results['documents'],
                metadatas=new_metadatas
            )
        chunk_count = len(new_ids)
        elapsed = time.perf_counter() - start
        logger.debug("Created %d vectors for user %s (reused) time=%.3fs", chunk_count, user_id, elapsed)
        return chunk_count

    def _extract_text_from_pdf(self, file_path: str) -> list[PageEntry]:
        """Extract text from a PDF file"""
        doc = None
        try:
            doc = pymupdf.open(file_path)
            pages = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text").strip()
                if len(text) > 10:  # ignore empty pages
                    pages.append(PageEntry(page_num=page_num, text=text))
            return pages
        finally:
            if doc is not None:
                doc.close()

    def _add_to_db(self, file_path: str, pages: list[PageEntry], user_id: str, file_hash: str, original_file_name: str):
        # Split text into smart chunks
        # Chunks per page -> embeds -> stores chunk + page metadata."""
        # pages -> page aware chunking (with overlap tokens + chunk_size as 600 + metadata (chunk no, pageno etc) Rebuild the context around a result (show surrounding text)) 
        # -> embeddings -> store in chroma(vector db) -> semantic search

        # Collect ALL chunks first (before writing to DB)
        all_ids = []
        all_metadatas = []
        all_embeddings = []
        all_documents = []
        for page in pages:

            chunks = self.text_splitter.split_text(page.text)
            page_ids = []
            page_metadatas = []
            # chunks:i is added as metadate here, as it can Trace back to the original document position (page, section, chunk).
            # Rebuild the context around a result (show surrounding text).
            for i in range(len(chunks)):

                page_ids.append(f"{str(user_id)}_{original_file_name}_{page.page_num}_{i}")
                page_metadatas.append({
                    "user_id": str(user_id), # string as chromadb metadta doesn't acceppt uuid type
                    "file": original_file_name,
                    "file_hash": file_hash,
                    "chunk": i,
                    "page_num": page.page_num})

            # Step 2: Convert sentences into embeddings (vectors)
            # sentence_embeddings = self.model.encode(docs).tolist()  # Encodes all documents (or text chunks) at once in a single batch, used when the list docs is small 
            # Embed in batches (efficient)
            sentence_embeddings = self.model.encode(chunks, batch_size=16, show_progress_bar=False).tolist() # Encodes the input in smaller batches of 16 chunks at a time

            # collect all info
            all_ids.extend(page_ids)
            all_metadatas.extend(page_metadatas)
            all_embeddings.extend(sentence_embeddings)
            all_documents.extend(chunks)
            # for tracking
            # self.loaded_files[file_path] = len(chunks)
            # logger.debug("Indexed %d chunks from %s", len(chunks), page.page_num)

        # Add to Chroma (with metadata)
        # requires one list of IDs and one list of metadatas aligned with each document
        # Thread safe db add
        with FileSearch._lock:
            self.collection.add(
                ids= all_ids,
                documents=all_documents,
                embeddings=all_embeddings,
                metadatas=all_metadatas
            )
        logger.info("Finished indexing %s for user %s: chunks=%d", original_file_name, user_id, len(all_ids))
        return len(all_ids)  # return total chunks indexed for tracking

    def _extract_text_from_word(self, file_path: str) -> list[PageEntry]:
        """Extract text from word (.docx) file"""
        pages = []
        doc = Document(file_path)
        file_contents =  '\n'.join(para.text for para in doc.paragraphs if para.text.strip())
        pages.append(PageEntry(page_num=None, text=file_contents))
        return pages

    def _extract_text_from_text(self, file_path: str) -> list[PageEntry]:
        """Extract text from a .txt file"""
        pages = []
        with open (file_path, 'r', encoding='utf-8', errors='replace') as file:
            file_contents = file.read()
        pages.append(PageEntry(page_num=None, text=file_contents))
        return pages

    def _get_embedding(self, search_key: str) -> str:
        """Return cached embedding if available."""
        search_key = search_key.lower()
        if search_key in self.embedding_cache:
            logger.debug("Embedding cache hit for key=%s", search_key)
            return self.embedding_cache[search_key]
        t0 = time.perf_counter()
        embedding = self.model.encode([search_key]).tolist()
        self.embedding_cache[search_key] = embedding
        logger.debug("Computed embedding for key=%s time=%.0fms", search_key, (time.perf_counter()-t0)*1000)
        return embedding
        
    def __keyword_search(self, search_key:str, user_id: str) -> list:
        """Search for keyword in stored sentences."""
        t0 = time.perf_counter()
        # step4: query
        query_embedding = self._get_embedding(search_key)
        logger.debug("embed time=%.0fms", (time.perf_counter()-t0)*1000)
        # 7. Search in Chroma
        t1 = time.perf_counter()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS, # return top 5
            where={"user_id": user_id}
        )
        logger.debug("chroma query time=%.0fms", (time.perf_counter()-t1)*1000)
        filtered_results = []
        t2 = time.perf_counter()
        # filtering response based on cosine similarity score
        for doc, score, meta in zip(results['documents'][0],results['distances'][0], results["metadatas"][0]):
            if score >= THRESHOLD:
                filtered_results.append({"file": meta["file"], 'page': meta['page_num'], 'chunk': meta['chunk'], "score": score, "text": doc})
        logger.debug("filter time=%.0fms results=%d", (time.perf_counter()-t2)*1000, len(filtered_results))
        return filtered_results

    def delete_from_db(self, file_hash:str, user_id: str) -> None:
        """ delete entries from vector db """
        with FileSearch._lock:
            # Use $eq operator for each condition
            self.collection.delete(
                where={
                    "$and": [
                        {"file_hash": {"$eq": file_hash}},
                        {"user_id": {"$eq": str(user_id)}}
                    ]
                }
            )

    def keyword_search(self, query: str, user_id: str) -> None:
        """Generate an answer from retrieved context using Ollama (RAG)"""
        t_start = time.perf_counter()
        # Step 1: Get top semantic matches
        search_results = self.__keyword_search(query, user_id)

        if not search_results:
            logger.warning(f"No relevant context found for query: '{query}' and user_id: {user_id}")
            return "No relevant context found in the indexed documents.", []

        # Step 2: Build the context from top chunks
        # context = '\n\n'.join(data['text'] for data in search_results)
        # context = ''
        # for data in search_results:
        #     context += f'File: {data['file']} | Page: {data['page']} \n{data['text']} \n'
        # # # Step 3: Create a RAG-style prompt
        # # prompt = f"""You are a helpful assistant.
        # # Use ONLY the context below to answer the question accurately and concisely.
        # # If the answer is not contained in the context, say: "The context does not provide enough information."
        
        # # Context:
        # # {context}

        # # Question: {query}
        # # Answer:
        # # """

        # # Step 4: Call llm model
        context = ''
        for i, data in enumerate(search_results, 1):
            context += f'[{i}] File: {data["file"]}, Page: {data["page"]}\n{data["text"]}\n\n'

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": """You are a strict document-grounded answering system.

                RULES (follow exactly — no exceptions):
                1. Answer ONLY using facts explicitly stated in the provided context — zero outside knowledge
                2. Every sentence in your answer MUST be directly supported by a specific passage in the context
                3. Do NOT paraphrase, infer, speculate, or add any detail not present word-for-word in the context
                4. If the context does not contain enough information, respond ONLY with: 'Not found in the provided documents.'
                5. Keep answers to 2-4 sentences; cite sources inline as [File: X, Page: Y] after each claim"""},
                {'role': 'user', 'content': f'Context: \n {context}\n Question: {query}'}], model=LLM_MODEL, temperature=0.0,
        )
        elapsed = time.perf_counter() - t_start
        logger.info("keyword_search completed: user=%s query_len=%d matches=%d time=%.3fs", user_id, len(query), len(search_results), elapsed)
        return chat_completion.choices[0].message.content, search_results

    def hybrid_retrieve(self, query: str, user_id: str) -> list:
        """Perform hybrid retrieval (BM25 + semantic search) and return top results"""
        return self._hybrid_retrieve(query, user_id)
    
    async def hybrid_generate_stream(self, query: str, search_results: list) -> any:
        """Perform hybrid generate"""
        if not search_results:
            logger.warning(f"No relevant context found for query: '{query}'")
            yield "No relevant context found in the indexed documents.", []
            return
        start = time.perf_counter()
        context = ''
        for i, data in enumerate(search_results, 1):
            context += f'[{i}] File: {data["file"]}, Page: {data["page"]}\n{data["text"]}\n\n'

        messages = [
            {
                "role": "system",
                "content": """You are a strict document-grounded answering system.

                RULES (follow exactly — no exceptions):
                1. Answer ONLY using facts explicitly stated in the provided context — zero outside knowledge
                2. Every sentence in your answer MUST be directly supported by a specific passage in the context
                3. Do NOT paraphrase, infer, speculate, or add any detail not present word-for-word in the context
                4. If the context does not contain enough information, respond ONLY with: 'Not found in the provided documents.'
                5. Keep answers to 2-4 sentences; cite sources inline as [File: X, Page: Y] after each claim"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\nQuery: {query}"
            }
        ]
        stream = await asyncio.wait_for(
            self._async_client.chat.completions.create(messages=messages, temperature=0.0, model=LLM_MODEL, stream=True),
            timeout=30
        )
        # normal for works with synchronous iterables
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None and token != '':
                yield token
        elapsed = time.perf_counter() - start
        logger.info("hybrid_generate_stream finished: query_len=%d matches=%d time=%.3fs", len(query), len(search_results), elapsed)
    

    def hybrid_search(self, query: str, user_id: str) -> tuple:
        """Generate an answer from retrieved context using Ollama (RAG)"""
        # Step 1: Get top semantic matches
        start = time.perf_counter()
        logger.info("Starting hybrid search for query: '%s' and user_id: %s", query, user_id)
        search_results = self._hybrid_retrieve(query, user_id)
        if not search_results:
            logger.warning(f"No relevant context found for query: '{query}'")
            return "No relevant context found in the indexed documents.", []

        # Step 3: Call llm model
        context = ''
        for i, data in enumerate(search_results, 1):
            context += f'[{i}] File: {data["file"]}, Page: {data["page"]}\n{data["text"]}\n\n'

        messages = [
            {
                "role": "system",
                "content": """You are a strict document-grounded answering system.

                RULES (follow exactly — no exceptions):
                1. Answer ONLY using facts explicitly stated in the provided context — zero outside knowledge
                2. Every sentence in your answer MUST be directly supported by a specific passage in the context
                3. Do NOT paraphrase, infer, speculate, or add any detail not present word-for-word in the context
                4. If the context does not contain enough information, respond ONLY with: 'Not found in the provided documents.'
                5. Keep answers to 2-4 sentences; cite sources inline as [File: X, Page: Y] after each claim"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\nQuery: {query}"
            }
        ]
        chat_completion = self.groq_client.chat.completions.create(messages=messages, temperature=0.0, model=LLM_MODEL)
        elapsed = time.perf_counter() - start
        logger.info("hybrid_search completed: user=%s query_len=%d matches=%d time=%.3fs", user_id, len(query), len(search_results), elapsed)
        return chat_completion.choices[0].message.content, search_results

    def _rebuild_bm25_for_user(self, user_id: str) -> None:
        results = self.collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        if not results['ids']:
            return
        self.bm25_ids[user_id] = results['ids']  # store the chunk ids for the user
        self.bm25_chunks[user_id] = results['documents']  # store the chunks for the user
        corpus = [doc.lower().split() for doc in results['documents']]
        self.bm25_indexes[user_id] = BM25Okapi(corpus)  # build BM25 index for the user's documents

    def _hybrid_retrieve(self, query: str, user_id: str, top_k: int = 10) -> list:
        """Build BM25 index for the user's documents (for keyword search)"""

        """
        RRF (Reciprocal Rank Fusion) is a method for combining multiple ranked lists (e.g., from different search algorithms). 
        It assigns scores to items based on their ranks in each list, giving higher scores to items that appear near the top of any list. The formula for RRF is:

        rrf ignores the actual scores from individual algorithms and focuses on the ranks. 
        The constant (e.g., 60) is a tuning parameter that controls how much influence lower-ranked items have on the final score. A smaller constant gives more weight to top-ranked items, while a larger constant allows lower-ranked items to contribute more to the final ranking.
        
        # The formula
        rrf_score = 1 / (k + rank)

        # For each chunk, sum its RRF score from both lists
        final_score = 1/(k + bm25_rank) + 1/(k + semantic_rank)

        eg:
        chunk_4:
        BM25 rank = 1      → 1/(60+1) = 0.01639
        Semantic rank = 2  → 1/(60+2) = 0.01613
        Final RRF = 0.03252  ✅ high — both agree it's good

        chunk_7:
        BM25 rank = 2      → 1/(60+2) = 0.01613
        Semantic rank = 8  → 1/(60+8) = 0.01471
        Final RRF = 0.03084

        chunk_9:
        BM25 rank = 15     → 1/(60+15) = 0.01333
        Semantic rank = 1  → 1/(60+1)  = 0.01639
        Final RRF = 0.02972

        # chunk_4 wins — both systems agreed
        # chunk_9 is still included — semantic liked it even though BM25 didn't
        """
        start = time.perf_counter()
        logger.debug("Starting hybrid retrieval for query: %s", query)
        # bm25
        fetch_k = top_k * 4  # fetch 40 if you want top 5 final results (ce output)
        query_tokens = query.lower().split()

        if user_id not in self.bm25_indexes:
            self._rebuild_bm25_for_user(user_id)  # build index if doesn't exists

        # if still empty after rebuild
        if user_id not in self.bm25_indexes:
            return []
        bm25_scores = self.bm25_indexes[user_id].get_scores(query_tokens)  # get BM25 scores for the query against all chunks

        # Get top-k BM25 results (returns list of tuples (chunk index, score)), highest first
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:fetch_k]

        # semantic search
        # query
        query_embedding = self._get_embedding(query)
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k, # return top 20
            where={"user_id": user_id}, include=["documents", "metadatas", "distances"]
        )
        # get top-k semantic results
        semantic_ids = results['ids'][0]  # top 20 chunk ids

        # RRF (reciprocal rank fusion)
        # Create a dictionary to hold the combined scores
        rrf_scores = {}
        # BM25 contributions
        for rank, (chunk_idx, _) in enumerate(bm25_ranked):
            chunk_id = self.bm25_ids[user_id][chunk_idx]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank + 1)

        # Semantic contributions
        for rank, chunk_id in enumerate(semantic_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank + 1)
            #                                        ↑
            #              .get(chunk_id, 0) handles the missing case
            #              if chunk only in semantic → starts from 0, adds semantic score
            #              if chunk in both → adds on top of existing BM25 score

        # Final ranking
        final_ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        results = self.collection.get(ids=final_ranked_ids, include=["documents", "metadatas"])

        rrf_results = []
        # filtering response based on cosine similarity score
        for id, doc, meta in zip(results['ids'],results['documents'], results["metadatas"]):
            rrf_results.append(
                {
                    "file": meta["file"],
                    'page': meta['page_num'],
                    'chunk': meta['chunk'],
                    "score": rrf_scores[id],
                    "text": doc
                }
            )
        # cross encoder 10 -> 5
        reranked = self._rerank(query, rrf_results, top_k=5)
        elapsed = time.perf_counter() - start
        logger.debug("Hybrid retrieve finished: query=%s returned=%d time=%.3fs", query, len(reranked), elapsed)
        return reranked

    def _rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank chunks using CrossEncoder — call after RRF, it tries to rerank the top results from RRF to improve relevance.
        """
        start = time.perf_counter()
        logger.debug("Reranking chunks with CrossEncoder... count=%d", len(chunks))
        if not chunks:
            return []
        # def trim(text: str, max_words: int = 350) -> str:
        #     # trim chunk text before passing to CE (CE has max token limit of 512, we can assume 1 word ~ 1.5 tokens on average, so 350 words ~ 525 tokens, leaving room for query and prompt)
        #     words = text.split()
        #     return ' '.join(words[:max_words]) if len(words) > max_words else text
        # Build (query, chunk text) pairs
        pairs = [(query, chunk['text']) for chunk in chunks]
        # score all the pairs, basically it predicts the relevance of each chunk to the query
        ce_scores = self.reranker.predict(pairs)
        # Attach ce score to each chunk
        for i in range(len(chunks)):
            chunks[i]['score'] = float(ce_scores[i])  # replaces original RRF score with CE score for final ranking, you can keep both if you want to analyze later
        # Sort by score and return top_k, filter by weak results
        #  direct questions → high scores, theme queries    → low scores 
        # reranked =  sorted(chunks, key=lambda x: x['score'], reverse=True)[:top_k]
        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
        # return top 1 as a fall back even if all the scores are below threshold
        reranked = [c for c in sorted_chunks[:top_k] if c['score'] >= CE_THRESHOLD] or sorted_chunks[:1]
        elapsed = time.perf_counter() - start
        logger.debug("Rerank completed: returned=%d time=%.3fs", len(reranked), elapsed)
        return reranked

    # def list_files(self):
    #     """List all loaded files"""
    #     return self.file_tracker.list_files()

if __name__ == '__main__':
    obj = FileSearch()
