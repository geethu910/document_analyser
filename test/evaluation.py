import json
import os
from dotenv import load_dotenv
import pandas as pd
from ragas import evaluate
# from ragas.dataset import Dataset
from datasets import Dataset
from groq import Groq
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from document_assistant.file_search_embeddings import FileSearch
from sentence_transformers import SentenceTransformer
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# # LangChain models
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
EMBED_MODEL = os.getenv('EMBED_MODEL', '')

def run_ragas_eval(eval_file):
    # embed_model = SentenceTransformer(EMBED_MODEL)
    # llm_model = Groq(api_key=GROQ_API_KEY)
    # Load test cases
    with open(eval_file, "r") as f:
        dataset = json.load(f)
    print('loaded dataset')
    fs = FileSearch()

    # Build RAGAS dataset (LIST OF DICTS)
    results = []
    
    for case in dataset:
        query = case["query"]
        expected = case["expected_answer"]

        # Retrieval (contexts)
        retrieved = fs._FileSearch__keyword_Search(query)
        contexts = [c["text"] for c in retrieved] if retrieved else [""]
        
        # Generation
        answer = fs.keyword_Search(query)

        results.append({
            "question": query,
            "ground_truth": expected,
            "answer": answer,
            "contexts": contexts
        })

    # df = pd.DataFrame(results)
    # ragas_dataset = Dataset.from_pandas(df)
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    # Wrap LLM 
    llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL)
    )
    # # Wrap Embeddings (MANDATORY)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name = EMBED_MODEL)
    )
    ragas_dataset = Dataset.from_list(results)
    print('calculating score !!!!')
    scores = evaluate(dataset= ragas_dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    # Save outputs
    # print('saving score !!!!')
    # with open("ragas_scores.json", "w") as f:
    #     json.dump(scores.to_dict(), f, indent=2)

    # print("\n====== RAGAS Scores =======")
    print(scores)


if __name__ == "__main__":
    file_path = '/home/geethuthomas/python_samples/document_analyser/test/eval_file.json'
    run_ragas_eval(file_path)

    # {'context_precision': 0.7167, 'context_recall': 1.0000, 'faithfulness': 0.6250, 'answer_relevancy': 0.6473}
    
    #
    """
    context_recall: Is the information needed to answer the question present somewhere in the retrieved chunks? --? Your retriever is not missing answers
    ✅ High score means

        Your retriever did not miss the answer
        The correct chunk exists in top-K results
    ❌ Low score means

        Wrong chunks retrieved
        Chunking/indexing issues
        k too small
    
    context_precision : 

    | Metric                | Question it answers                            |
| --------------------- | ---------------------------------------------- |
| **Context Recall**    | *Did we retrieve the right info at all?*       |
| **Context Precision** | *How much of what we retrieved is useful?*     |
| **Faithfulness**      | *Did the answer stick to the retrieved text?*  |
| **Answer Relevancy**  | *Did the answer actually answer the question?* |


    """
