# ragas_eval.py
import os
import pandas as pd
from dotenv import load_dotenv
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.llms.base import LangchainLLMWrapper as RealWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import httpx


GROUND_TRUTH_PATH = "eval_file.json"
BASE_URL = "http://localhost:8000"
TOKEN = ""  # replace with bearer token
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
EMBED_MODEL = os.getenv('EMBED_MODEL', '')


class GroqLLMWrapper(RealWrapper):
    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return super().generate_text(prompt, n=1, temperature=temperature,
                                     stop=stop, callbacks=callbacks)
    
    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return await super().agenerate_text(prompt, n=1, temperature=temperature,
                                            stop=stop, callbacks=callbacks)

def run_search(query: str, mode: str) -> tuple[str, list[str]]:
    """Call your API and return (answer, context_chunks)"""
    endpoint = "/api/hybrid_search" if mode == "hybrid" else "/api/search"
    try:
        res = httpx.post(
            f"{BASE_URL}{endpoint}",
            params={"query": query},
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=60
        )
    except Exception as e:
        raise RuntimeError(f"HTTP request failed for query {query!r}: {e}")

    # Try to parse JSON, but surface raw text on failure
    try:
        data = res.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON response ({res.status_code}) for query {query!r}: {res.text!r}")

    # Debug info: show status and keys
    print(f'### Data for query: "{query}", status: {res.status_code}, data: {list(data.keys())}')

    # If the API returned an error, include the detail in the exception so it's visible
    if res.status_code != 200:
        detail = data.get("detail", res.text)
        raise RuntimeError(f"Search API returned status {res.status_code}: {detail}")

    # Defensive access to expected fields
    if "answer" not in data:
        raise KeyError(f"'answer' missing in response for query {query!r}. Response keys: {list(data.keys())}. Full response: {data}")

    answer = data["answer"]
    contexts = [m.get("text", "") for m in data.get("matches", [])]
    return answer, contexts

def build_ragas_dataset(ground_truth: list, mode: str) -> Dataset:
    rows = []
    for item in ground_truth:
        answer, contexts = run_search(item["question"], mode)
        rows.append({
            "question":   item["question"],
            "answer":     answer,
            "contexts":   contexts,
            "ground_truth": item["ground_truth"]
        })
    return Dataset.from_list(rows)

def unwrap(x):
    return x[0] if isinstance(x, list) else x


def main():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    metrics = [faithfulness, answer_relevancy, context_recall]
    for m in metrics:
        if hasattr(m, '_reproducibility'):
            m._reproducibility = 1
            print(f"Patched {m.name}: _reproducibility = 1")
    # RAGAS needs an LLM for evaluation — use Groq
    # groq_llm = LangchainLLMWrapper(
    #     ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, request_timeout=60, max_retries=3, temperature=0, n=1)
    # )
    groq_llm = GroqLLMWrapper(
        ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, request_timeout=60, max_retries=3, temperature=0)
    )

    # for metric in metrics:
        # metric.llm = groq_llm

    #  Wrap Embeddings (MANDATORY)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    results = {}
    for mode in ["semantic", "hybrid"]:
        print(f"\nBuild ragas dataset {mode}...")
        dataset = build_ragas_dataset(ground_truth, mode)
        print(f'######Evaluating {mode}...')
        score = evaluate(dataset, metrics=metrics, llm=groq_llm, embeddings=embeddings,
                         run_config=RunConfig(max_workers=1, timeout=180, max_retries=5), batch_size=1)
        print(f"{mode}: {score}")

        df = score.to_pandas()
        df.to_csv(f"{mode}_results.csv", index=False)
        summary = df[['faithfulness', 'answer_relevancy', 'context_recall']].describe()
        summary.to_csv(f"{mode}_summary.csv")
        print(f"\n--- {mode} NaN report ({len(df)} total samples) ---")
        for metric in ['faithfulness', 'answer_relevancy', 'context_recall']:
            nan_count = df[metric].isna().sum()
            valid = len(df) - nan_count
            avg = df[metric].mean()  # already skipna=True
            print(f"  {metric:<20}: {valid:>2}/{len(df)} scored | avg = {avg:.4f}" + 
                (f" ⚠️  {nan_count} NaN" if nan_count > 0 else " ✅"))
        results[mode] = df[['faithfulness', 'answer_relevancy', 'context_recall']].mean().to_dict()
        

    # Print comparison table
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]

    print("\n" + "="*54)
    print(f"| {'Metric':<20} | {'Semantic':>8} | {'Hybrid':>8} |")
    print("|" + "-"*22 + "|" + "-"*10 + "|" + "-"*10 + "|")
    for metric in metrics:
        s = float(results["semantic"][metric])
        h = float(results["hybrid"][metric])
        print(f"| {metric:<20} | {s:>8.4f} | {h:>8.4f} |")
    print("="*54)

if __name__ == "__main__":
    main()