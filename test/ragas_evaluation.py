# ragas_eval.py
import os
import re
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
# Use a STRONG, separate judge model for RAGAS. Faithfulness/answer_relevancy are LLM-judged;
# a small generator model makes a harsh/unreliable judge and deflates scores. Override via env.
JUDGE_LLM_MODEL = os.getenv('JUDGE_LLM_MODEL', 'llama-3.3-70b-versatile')
EMBED_MODEL = os.getenv('EMBED_MODEL', '')

# Matches inline citations like "[File: advs.pdf, Page: 12]" that the answering prompt injects.
# These add noise that lowers answer_relevancy, so strip them before scoring.
_CITATION_RE = re.compile(r"\[File:[^\]]*\]")


def strip_citations(text: str) -> str:
    """Remove inline [File: X, Page: Y] citations and collapse leftover whitespace."""
    if not text:
        return text
    return re.sub(r"\s{2,}", " ", _CITATION_RE.sub("", text)).strip()


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

    # Strip inline citations
    answer = strip_citations(data["answer"])
    # Prefer the untruncated chunk (text_full); fall back to the truncated preview for older responses.
    # RAGAS faithfulness/context_recall compare against these contexts, so they MUST match what the
    # LLM actually read — truncated 300-char previews were making grounded answers look unfaithful.
    contexts = [m.get("text_full") or m.get("text", "") for m in data.get("matches", [])]
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
    # RAGAS needs an LLM for evaluation. Use a STRONG judge model (JUDGE_LLM_MODEL), separate from
    # the generator, so faithfulness/answer_relevancy verdicts are reliable. n=1 is required by Groq.
    groq_llm = GroqLLMWrapper(
        ChatGroq(model=JUDGE_LLM_MODEL, api_key=GROQ_API_KEY, request_timeout=60, max_retries=3, temperature=0)
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