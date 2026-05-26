"""
eval_hybrid.py — Hybrid Search Evaluation Script
Run from project root: python eval_hybrid.py --user_id <your_user_id>

What this does:
  - Runs 20 queries against hybrid_search() and keyword_search()
  - Records scores, ranks, timing, answer quality
  - Saves results to eval_results.json + eval_results.csv
  - Prints a summary table to terminal
"""

import json
import time
import argparse
import csv
from datetime import datetime
from document_assistant.file_search_embeddings import FileSearch

# ── CONFIG ────────────────────────────────────────────────────────────────────

QUERIES = [
    # (query, type, expected_file_hint)
    # type: "direct" = has a direct answer, "theme" = topic query,
    #       "cross" = should hit multiple files, "keyword" = BM25 should dominate,
    #       "negative" = should return nothing relevant

    # Direct questions — expect CE scores > 0.5
    ("why did Charlotte Lucas marry Mr Collins",        "direct",   "Pride.pdf"),
    ("what does Sherlock Holmes deduce from observation","direct",   "advs.pdf"),
    ("who is Irene Adler",                              "direct",   "advs.pdf"),
    ("what happened to Hosmer Angel",                   "direct",   "advs.pdf"),
    ("how does Holmes identify criminals",              "direct",   "advs.pdf"),

    # Theme queries — expect low CE scores but correct chunks
    ("class and social status in marriage",             "theme",    "Pride.pdf"),
    ("watson and holmes friendship",                    "theme",    "advs.pdf"),
    ("women and independence in Victorian society",     "theme",    "Pride.pdf"),
    ("deception and hidden identity",                   "theme",    "both"),
    ("detective observation skills",                    "theme",    "advs.pdf"),

    # Cross-file queries — should hit multiple files
    ("brilliant woman who outsmarts men",               "cross",    "both"),
    ("a man who misjudges someone",                     "cross",    "both"),
    ("marriage for money not love",                     "cross",    "Pride.pdf"),
    ("woman with sharp mind",                           "cross",    "both"),

    # Keyword exact match — BM25 should dominate
    ("Mr Darcy Netherfield",                            "keyword",  "Pride.pdf"),
    ("Sherlock Holmes Baker Street",                    "keyword",  "advs.pdf"),
    ("Charlotte Lucas Mr Collins",                      "keyword",  "Pride.pdf"),
    ("Irene Adler photograph",                          "keyword",  "advs.pdf"),

    # Negative — should return empty or very low scores
    ("machine learning neural networks",                "negative", "none"),
    ("blockchain cryptocurrency bitcoin",               "negative", "none"),
]

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(user_id: str, compare_semantic: bool = True):
    print(f"\n{'='*60}")
    print(f"  Hybrid Search Evaluation")
    print(f"  User ID: {user_id}")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    fs = FileSearch()
    results = []

    for i, (query, qtype, expected_file) in enumerate(QUERIES):
        print(f"[{i+1:02d}/{len(QUERIES)}] {qtype.upper():8s} | {query}")

        row = {
            "query":          query,
            "type":           qtype,
            "expected_file":  expected_file,
        }

        # ── Hybrid search ──────────────────────────────────────────────────
        t0 = time.time()
        try:
            answer_h, chunks_h = fs.hybrid_search(query, user_id)
            hybrid_time = (time.time() - t0) * 1000

            top_score    = chunks_h[0]['score']   if chunks_h else 0
            top_file     = chunks_h[0]['file']    if chunks_h else "none"
            top_page     = chunks_h[0]['page']    if chunks_h else "-"
            files_found  = list({c['file'] for c in chunks_h})
            match_count  = len(chunks_h)

            row.update({
                "hybrid_answer":      answer_h,
                "hybrid_top_score":   round(top_score, 6),
                "hybrid_top_file":    top_file,
                "hybrid_top_page":    top_page,
                "hybrid_files_found": files_found,
                "hybrid_match_count": match_count,
                "hybrid_time_ms":     round(hybrid_time, 1),
                "hybrid_all_chunks":  [
                    {"file": c['file'], "page": c['page'], "score": round(c['score'], 6)}
                    for c in chunks_h
                ],
            })

            # Auto-grade: did rank 1 come from expected file?
            if expected_file == "none":
                grade = "✅" if top_score < 0.05 else "⚠️ "
            elif expected_file == "both":
                grade = "✅" if len(files_found) > 1 else "⚠️ "
            else:
                grade = "✅" if expected_file in top_file else "❌"

            row["hybrid_grade"] = grade

            print(f"          hybrid  → {grade} rank1: {top_file} p{top_page} "
                  f"score={top_score:.5f} ({hybrid_time:.0f}ms) chunks={match_count}")

        except Exception as e:
            row.update({
                "hybrid_answer": f"ERROR: {e}",
                "hybrid_top_score": 0,
                "hybrid_top_file": "error",
                "hybrid_grade": "💥",
                "hybrid_time_ms": 0,
            })
            print(f"          hybrid  → 💥 ERROR: {e}")

        # ── Semantic search (comparison) ───────────────────────────────────
        if compare_semantic:
            t0 = time.time()
            try:
                answer_s, chunks_s = fs.keyword_search(query, user_id)
                semantic_time = (time.time() - t0) * 1000

                s_top_score = chunks_s[0]['score'] if chunks_s else 0
                s_top_file  = chunks_s[0]['file']  if chunks_s else "none"
                s_top_page  = chunks_s[0]['page']  if chunks_s else "-"

                row.update({
                    "semantic_top_score": round(s_top_score, 6),
                    "semantic_top_file":  s_top_file,
                    "semantic_top_page":  s_top_page,
                    "semantic_time_ms":   round(semantic_time, 1),
                })

                # Did hybrid beat semantic on rank 1?
                if expected_file not in ("none", "both"):
                    h_correct = expected_file in row.get("hybrid_top_file", "")
                    s_correct = expected_file in s_top_file
                    if h_correct and not s_correct:
                        winner = "hybrid wins 🏆"
                    elif s_correct and not h_correct:
                        winner = "semantic wins"
                    elif h_correct and s_correct:
                        winner = "both correct"
                    else:
                        winner = "both wrong"
                    row["winner"] = winner
                    print(f"          semantic→    rank1: {s_top_file} p{s_top_page} "
                          f"score={s_top_score:.5f} ({semantic_time:.0f}ms) | {winner}")
                else:
                    print(f"          semantic→    rank1: {s_top_file} p{s_top_page} "
                          f"score={s_top_score:.5f} ({semantic_time:.0f}ms)")

            except Exception as e:
                row["semantic_top_score"] = 0
                row["semantic_top_file"]  = "error"
                print(f"          semantic→ 💥 ERROR: {e}")

        print()
        results.append(row)

    # ── Save results ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON — everything
    json_path = f"eval_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV — summary table
    csv_path = f"eval_results_{timestamp}.csv"
    csv_fields = [
        "query", "type", "expected_file",
        "hybrid_grade", "hybrid_top_file", "hybrid_top_page",
        "hybrid_top_score", "hybrid_time_ms", "hybrid_match_count",
        "semantic_top_file", "semantic_top_page", "semantic_top_score",
        "semantic_time_ms", "winner"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    # By query type
    for qtype in ["direct", "theme", "cross", "keyword", "negative"]:
        type_rows = [r for r in results if r["type"] == qtype]
        correct   = sum(1 for r in type_rows if r.get("hybrid_grade") == "✅")
        print(f"  {qtype.upper():8s}  {correct}/{len(type_rows)} correct")

    total_correct = sum(1 for r in results if r.get("hybrid_grade") == "✅")
    print(f"\n  TOTAL     {total_correct}/{len(results)} correct")

    # Timing
    hybrid_times   = [r["hybrid_time_ms"] for r in results if "hybrid_time_ms" in r]
    semantic_times = [r.get("semantic_time_ms", 0) for r in results if r.get("semantic_time_ms")]
    if hybrid_times:
        print(f"\n  Hybrid avg latency:   {sum(hybrid_times)/len(hybrid_times):.0f}ms")
    if semantic_times:
        print(f"  Semantic avg latency: {sum(semantic_times)/len(semantic_times):.0f}ms")

    # Winner breakdown
    winners = [r.get("winner", "") for r in results]
    print(f"\n  Hybrid wins:    {winners.count('hybrid wins 🏆')}")
    print(f"  Both correct:   {winners.count('both correct')}")
    print(f"  Semantic wins:  {winners.count('semantic wins')}")
    print(f"  Both wrong:     {winners.count('both wrong')}")

    # CE score analysis by type
    print(f"\n  CE Score Analysis:")
    for qtype in ["direct", "theme", "keyword"]:
        type_rows  = [r for r in results if r["type"] == qtype and "hybrid_top_score" in r]
        avg_score  = sum(r["hybrid_top_score"] for r in type_rows) / len(type_rows) if type_rows else 0
        print(f"  {qtype.upper():8s}  avg top CE score: {avg_score:.5f}")

    print(f"\n  Results saved to:")
    print(f"  → {json_path}  (full details)")
    print(f"  → {csv_path}   (summary table, open in Excel/Sheets)")
    print(f"{'='*60}\n")

    return results


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hybrid search quality")
    parser.add_argument(
        "--user_id",
        required=True,
        help="Your user ID (UUID from users table)"
    )
    parser.add_argument(
        "--no_compare",
        action="store_true",
        help="Skip semantic search comparison (faster)"
    )
    args = parser.parse_args()

    evaluate(
        user_id=args.user_id,
        compare_semantic=not args.no_compare
    )