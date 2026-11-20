"""
RegForg dashboard.

    python app.py            ->  http://127.0.0.1:7860

Tabs
  Ask                  streaming chat with cited sources and per-stage latency
  Retrieval Inspector  see exactly what the retriever returns, without the LLM
  Evaluation           run the eval set and inspect every question
  Monitoring           latency and refusal trends from the query log
  Knowledge Base       system health, indexed documents, upload and re-index
"""

import html
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import gradio as gr
import pandas as pd

from config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_THINKING,
    OLLAMA_BASE_URL,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from evaluation.evaluate import load_latest, run_evaluation
from evaluation.metrics import mean, percentile
from src.ingestion import indexer
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.pipeline.rag_pipeline import stream_rag
from src.retrieval.retriever import get_retriever
from src.utils.query_log import clear_queries, read_queries

EXAMPLES = [
    "What is the difference between supervised and unsupervised learning?",
    "How does gradient descent work?",
    "What is the kernel trick in SVMs?",
    "How do L1 and L2 regularization differ?",
    "How does a random forest work?",
    "Who won the 2018 FIFA World Cup?",
]

CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
#rf-header { padding: 20px 4px 4px; }
#rf-header h1 { font-size: 28px; font-weight: 750; letter-spacing: -0.02em; margin: 0; }
#rf-header p { color: var(--body-text-color-subdued); margin: 6px 0 0; font-size: 14px; }
.rf-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.rf-pill { font-size: 12px; padding: 3px 10px; border-radius: 999px;
           border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.rf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
.rf-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
           background: var(--block-background-fill); }
.rf-card .v { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.2; }
.rf-card .l { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
              color: var(--body-text-color-subdued); margin-top: 4px; }
.rf-card .s { font-size: 12px; color: var(--body-text-color-subdued); margin-top: 2px; }
.rf-good .v { color: #16a34a; } .rf-warn .v { color: #d97706; } .rf-bad .v { color: #dc2626; }
.rf-source { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
             margin-bottom: 10px; background: var(--block-background-fill); }
.rf-source.cited { border-left: 3px solid var(--color-accent); }
.rf-source .h { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; font-weight: 600; }
.rf-source .h a { color: var(--color-accent); text-decoration: none; }
.rf-source .t { font-size: 13px; line-height: 1.55; color: var(--body-text-color-subdued); margin-top: 8px;
                max-height: 9.5em; overflow: auto; }
.rf-bar { height: 4px; border-radius: 2px; background: var(--border-color-primary); margin-top: 8px; overflow: hidden; }
.rf-bar > div { height: 100%; background: var(--color-accent); }
.rf-muted { color: var(--body-text-color-subdued); font-size: 13px; padding: 8px 2px; }
.rf-badge { font-size: 11px; padding: 1px 8px; border-radius: 999px; font-weight: 600; }
.rf-badge.ok { background: rgba(22,163,74,.12); color: #16a34a; }
.rf-badge.no { background: rgba(217,119,6,.14); color: #d97706; }
.rf-badge.err { background: rgba(220,38,38,.12); color: #dc2626; }
"""


# ------------------------------------------------------------------
# Small HTML helpers
# ------------------------------------------------------------------

def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def secs(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}s"


def pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def card(value: str, label: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="s">{esc(sub)}</div>' if sub else ""
    return f'<div class="rf-card {tone}"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>{sub_html}</div>'


def grid(cards: List[str]) -> str:
    return f'<div class="rf-grid">{"".join(cards)}</div>'


def muted(text: str) -> str:
    return f'<div class="rf-muted">{esc(text)}</div>'


def pdf_link(file_name: str, page: int) -> str:
    path = (DOCUMENTS_DIR / file_name).resolve()
    if not path.exists():
        return f"{esc(file_name)} · p.{page}"
    return f'<a href="/gradio_api/file={quote(str(path))}#page={page}" target="_blank">{esc(file_name)} · p.{page} ↗</a>'


def tone_for(value: Optional[float], good: float, warn: float, higher_is_better: bool = True) -> str:
    if value is None:
        return ""
    if not higher_is_better:
        value, good, warn = -value, -good, -warn
    return "rf-good" if value >= good else "rf-warn" if value >= warn else "rf-bad"


# ------------------------------------------------------------------
# Ask tab
# ------------------------------------------------------------------

def sources_html(sources: List[Dict[str, Any]], cited: List[int], matches: List[Dict[str, Any]], min_score: float) -> str:
    if not matches:
        return muted("Retrieved sources appear here.")

    if not sources:
        best = matches[0]
        return muted(
            f"No chunk scored above the relevance threshold ({min_score:.2f}). "
            f"Best match was {best['score']:.3f} on page {best['page']}, so the question "
            f"was refused without calling the LLM."
        )

    blocks = []
    for i, s in enumerate(sources, start=1):
        is_cited = i in cited
        badge = '<span class="rf-badge ok">cited</span>' if is_cited else ""
        blocks.append(
            f'<div class="rf-source {"cited" if is_cited else ""}">'
            f'<div class="h"><span>[{i}] {pdf_link(s["file_name"], s["page"])} {badge}</span>'
            f'<span>{s["score"]:.3f}</span></div>'
            f'<div class="rf-bar"><div style="width:{max(0, min(1, s["score"])) * 100:.0f}%"></div></div>'
            f'<div class="t">{esc(s["text"])}</div></div>'
        )
    return "".join(blocks)


def answer_metrics_html(result: Optional[Dict[str, Any]], pending: str = "") -> str:
    if result is None:
        return muted(pending or "Latency and retrieval metrics for each answer appear here.")

    if result.get("gated"):
        status = card("Refused", "Status", "below relevance threshold", "rf-warn")
    elif result.get("abstained"):
        status = card("Refused", "Status", "model found no answer", "rf-warn")
    else:
        status = card("Answered", "Status", f"{len(result['cited'])}/{len(result['sources'])} sources cited", "rf-good")

    return grid([
        status,
        card(f"{result['top_score']:.3f}", "Top score", f"threshold {result['min_score']:.2f}"),
        card(secs(result["retrieval_time"]), "Retrieval", f"embed {secs(result['embedding_time'])} · search {secs(result['search_time'])}"),
        card(secs(result["time_to_first_token"]), "First token"),
        card(secs(result["generation_time"]), "Generation"),
        card(secs(result["total_time"]), "Total"),
    ])


def ask(message: str, history: List[Dict[str, str]], top_k: int, min_score: float):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        yield history, "", gr.skip(), gr.skip()
        return

    history += [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield history, "", answer_metrics_html(None, "Retrieving…"), muted("Searching the knowledge base…")

    try:
        for event in stream_rag(message, top_k=int(top_k), min_score=float(min_score)):
            if event["type"] == "retrieval":
                pending = f"Generating from {len(event['relevant'])} sources…" if event["relevant"] else "Refusing…"
                yield history, "", answer_metrics_html(None, pending), sources_html(
                    event["relevant"], [], event["matches"], float(min_score)
                )
            elif event["type"] == "token":
                history[-1]["content"] += event["text"]
                yield history, "", gr.skip(), gr.skip()
            else:
                history[-1]["content"] = event["answer"]
                yield history, "", answer_metrics_html(event), sources_html(
                    event["sources"], event["cited"], event["matches"], float(min_score)
                )
    except Exception as error:  # surface the cause instead of a generic error
        history[-1]["content"] = f"⚠️ {type(error).__name__}: {error}\n\nCheck the Knowledge Base tab for system health."
        yield history, "", muted("Request failed."), muted("")


# ------------------------------------------------------------------
# Retrieval inspector
# ------------------------------------------------------------------

def inspect_retrieval(query: str, top_k: int, min_score: float):
    if not (query or "").strip():
        return pd.DataFrame(), None, muted("Enter a query.")

    try:
        result = get_retriever().retrieve(query, top_k=int(top_k), min_score=float(min_score))
    except Exception as error:  # show the cause instead of failing silently
        return pd.DataFrame(), None, muted(
            f"⚠️ {type(error).__name__}: {esc(str(error))} — check the Knowledge Base tab for system health."
        )
    rows = [
        {
            "rank": m["rank"],
            "score": round(m["score"], 4),
            "passes threshold": "yes" if m["score"] >= min_score else "no",
            "page": m["page"],
            "text": m["text"][:240] + ("…" if len(m["text"]) > 240 else ""),
        }
        for m in result["matches"]
    ]
    table = pd.DataFrame(rows)
    chart = pd.DataFrame({
        "rank": [f"#{r['rank']} p.{r['page']}" for r in rows],
        "score": [r["score"] for r in rows],
        "passes threshold": [r["passes threshold"] for r in rows],
    })
    summary = grid([
        card(f"{result['top_score']:.3f}", "Top score"),
        card(f"{len(result['relevant'])}/{len(rows)}", "Above threshold", f"{min_score:.2f}"),
        card(secs(result["embedding_time"]), "Embedding"),
        card(secs(result["search_time"]), "Vector search"),
    ])
    return table, chart, summary


# ------------------------------------------------------------------
# Evaluation tab
# ------------------------------------------------------------------

def eval_views(report: Optional[Dict[str, Any]]):
    if not report:
        empty = muted("No evaluation has been run yet. Click Run evaluation.")
        return empty, pd.DataFrame(), None, ""

    s, cfg = report["summary"], report["config"]
    cards = [
        card(f"{s['hit_at_k']:.2f}", f"Hit@{cfg['top_k']}", "relevant page retrieved", tone_for(s["hit_at_k"], .9, .75)),
        card(f"{s['mrr']:.2f}", "MRR", "rank of first relevant page", tone_for(s["mrr"], .8, .6)),
        card(f"{s['recall_at_k']:.2f}", f"Recall@{cfg['top_k']}", "share of relevant pages", tone_for(s["recall_at_k"], .8, .6)),
        card(pct(s["refusal_rate"]), "Off-topic refused", f"{cfg['n_unanswerable']} questions", tone_for(s["refusal_rate"], .9, .7)),
        card(pct(s["false_refusal_rate"]), "False refusals", f"{cfg['n_answerable']} questions", tone_for(s["false_refusal_rate"], .05, .15, False)),
    ]
    if cfg["generate"]:
        cards += [
            card(pct(s["citation_rate"]), "Answers cited", "valid [n] citation", tone_for(s["citation_rate"], .9, .7)),
            card(secs(s["generation_p50"]), "Generation p50", f"p95 {secs(s['generation_p95'])}"),
            card(secs(s["total_p50"]), "End-to-end p50", f"p95 {secs(s['total_p95'])}"),
        ]
    cards.append(card(secs(s["retrieval_p50"]), "Retrieval p50", f"p95 {secs(s['retrieval_p95'])}"))

    rows = []
    for i, r in enumerate(report["rows"], start=1):
        row = {
            "#": i,
            "question": r["question"],
            "type": "in-scope" if r["answerable"] else "out-of-scope",
            "top score": r["top_score"],
            "refused": "yes" if r["refused"] else "no",
            "MRR": round(r["mrr"], 2) if r["answerable"] else None,
            "expected pages": ", ".join(map(str, r["relevant_pages"])) or "—",
            "retrieved pages": ", ".join(map(str, r["retrieved_pages"])),
        }
        if cfg["generate"]:
            row["answer"] = r.get("answer", "")
        rows.append(row)

    chart = pd.DataFrame({
        "question #": [row["#"] for row in rows],
        "top score": [row["top score"] for row in rows],
        "type": [row["type"] for row in rows],
    })
    meta = (
        f"Run {esc(report['run_at'])} · top_k={cfg['top_k']} · threshold={cfg['min_score']} · "
        f"{'retrieval + generation' if cfg['generate'] else 'retrieval only'} · "
        f"{esc(cfg['embedding_model'])} / {esc(cfg['llm_model'])}"
    )
    return grid(cards), pd.DataFrame(rows), chart, f'<div class="rf-muted">{meta}</div>'


def run_eval(top_k: int, min_score: float, generate: bool, progress=gr.Progress()):
    report = run_evaluation(
        top_k=int(top_k), min_score=float(min_score), generate=bool(generate),
        progress=lambda f, msg: progress(f, desc=msg),
    )
    return eval_views(report)


# ------------------------------------------------------------------
# Monitoring tab
# ------------------------------------------------------------------

def monitoring_views():
    records = read_queries()
    if not records:
        return muted("No questions logged yet. Ask something in the Ask tab."), None, pd.DataFrame()

    total = [r["total_time"] for r in records]
    answered = [r for r in records if not r["abstained"]]
    cards = grid([
        card(str(len(records)), "Questions"),
        card(pct(mean(float(r["abstained"]) for r in records)), "Refused"),
        card(secs(percentile(total, 50)), "Total p50", f"p95 {secs(percentile(total, 95))}"),
        card(secs(percentile([r["time_to_first_token"] for r in answered], 50)), "First token p50"),
        card(secs(percentile([r["retrieval_time"] for r in records], 50)), "Retrieval p50"),
        card(f"{mean(r['top_score'] for r in records):.3f}", "Mean top score"),
    ])

    chart_rows = []
    for i, r in enumerate(records, start=1):
        chart_rows.append({"query #": i, "seconds": round(r["retrieval_time"], 3), "stage": "retrieval"})
        chart_rows.append({"query #": i, "seconds": round(r["generation_time"] or 0, 3), "stage": "generation"})

    table = pd.DataFrame([
        {
            "time (UTC)": r["timestamp"].replace("T", " ").replace("+00:00", ""),
            "question": r["query"],
            "status": "refused" if r["abstained"] else "answered",
            "top score": round(r["top_score"], 3),
            "retrieval s": round(r["retrieval_time"], 2),
            "generation s": round(r["generation_time"] or 0, 2),
            "total s": round(r["total_time"], 2),
            "pages": ", ".join(map(str, r.get("pages", []))),
        }
        for r in reversed(records)
    ])
    return cards, pd.DataFrame(chart_rows), table


def clear_monitoring():
    clear_queries()
    return monitoring_views()


# ------------------------------------------------------------------
# Knowledge base tab
# ------------------------------------------------------------------

def health_html() -> str:
    checks = []

    try:
        import ollama

        models = {m.model for m in ollama.Client(host=OLLAMA_BASE_URL).list().models}
        for name in (EMBEDDING_MODEL, LLM_MODEL):
            ok = name in models
            checks.append((f"Ollama model {name}", ok, "pulled" if ok else f"run: ollama pull {name}"))
    except Exception as error:
        checks.append(("Ollama server", False, f"not reachable at {OLLAMA_BASE_URL} ({type(error).__name__}). Run: ollama serve"))

    try:
        count = indexer.vector_count()
        checks.append((f"Pinecone {PINECONE_INDEX_NAME}/{PINECONE_NAMESPACE}", count > 0, f"{count} vectors"))
    except Exception as error:
        checks.append(("Pinecone", False, f"{type(error).__name__}: {error}"))

    rows = "".join(
        f'<div class="rf-source"><div class="h"><span>{esc(name)}</span>'
        f'<span class="rf-badge {"ok" if ok else "err"}">{"OK" if ok else "FAIL"}</span></div>'
        f'<div class="t">{esc(detail)}</div></div>'
        for name, ok, detail in checks
    )
    config_line = (
        f"Embedding {EMBEDDING_MODEL} ({PINECONE_DIMENSION}-d) · LLM {LLM_MODEL} "
        f"(thinking {'on' if LLM_THINKING else 'off'}) · default top_k {TOP_K} · threshold {RELEVANCE_THRESHOLD}"
    )
    return rows + muted(config_line)


def documents_table() -> pd.DataFrame:
    manifest = indexer.load_manifest()
    on_disk = {p.name for p in DOCUMENTS_DIR.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS}
    rows = []
    for name in sorted(on_disk | set(manifest)):
        entry = manifest.get(name, {})
        rows.append({
            "document": name,
            "status": "indexed" if entry else "not indexed",
            "pages": entry.get("pages"),
            "chunks": entry.get("chunks"),
            "indexed at (UTC)": (entry.get("indexed_at") or "").replace("T", " ").replace("+00:00", ""),
            "on disk": "yes" if name in on_disk else "no",
        })
    return pd.DataFrame(rows)


def kb_views():
    return health_html(), documents_table()


def upload_and_index(file_path: Optional[str], progress=gr.Progress()):
    if not file_path:
        return muted("Choose a file first."), health_html(), documents_table()

    source = Path(file_path)
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return muted(f"Unsupported type {source.suffix}."), health_html(), documents_table()

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCUMENTS_DIR / source.name
    shutil.copy(source, target)
    result = indexer.index_file(target, progress=lambda f, msg: progress(f, desc=msg))
    message = f"{result['file_name']}: {result['status']} · {result['chunks']} chunks"
    return muted(message), health_html(), documents_table()


def resync(progress=gr.Progress()):
    results = indexer.sync_all(progress=lambda f, msg: progress(f, desc=msg))
    message = " · ".join(f"{r['file_name']}: {r['status']}" for r in results)
    return muted(message), health_html(), documents_table()


def remove_document(name: str):
    name = (name or "").strip()
    if not name:
        return muted("Enter the document name to remove."), health_html(), documents_table()
    removed = indexer.remove_file(name)
    return muted(f"Removed {removed} vectors for {name} (the file on disk is kept)."), health_html(), documents_table()


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="RegForg") as demo:
        gr.HTML(
            f"""<div id="rf-header"><h1>RegForg</h1>
            <p>Private, local RAG over your documents. Every answer is grounded and cited, and off-topic questions are refused.</p>
            <div class="rf-pills"><span class="rf-pill">Ollama · {esc(LLM_MODEL)}</span>
            <span class="rf-pill">{esc(EMBEDDING_MODEL)} · {PINECONE_DIMENSION}-d</span>
            <span class="rf-pill">Pinecone · {esc(PINECONE_INDEX_NAME)}</span></div></div>"""
        )

        with gr.Tabs():
            # ---------------- Ask ----------------
            with gr.Tab("Ask"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=520,
                            show_label=False,
                            placeholder="Ask a question about your documents. Each question is answered independently from the knowledge base.",
                            latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                              {"left": "$", "right": "$", "display": False}],
                        )
                        with gr.Row():
                            question = gr.Textbox(placeholder="e.g. How does gradient descent work?", show_label=False, scale=6, autofocus=True)
                            send = gr.Button("Ask", variant="primary", scale=1, min_width=90)
                        gr.Examples(EXAMPLES, inputs=question, label="Examples")
                        with gr.Accordion("Retrieval settings", open=False):
                            top_k = gr.Slider(1, 15, value=TOP_K, step=1, label="Top-K chunks")
                            threshold = gr.Slider(0.0, 0.9, value=RELEVANCE_THRESHOLD, step=0.01, label="Relevance threshold (cosine)")
                        clear = gr.Button("Clear conversation", size="sm", variant="secondary")
                    with gr.Column(scale=2):
                        metrics = gr.HTML(answer_metrics_html(None))
                        gr.Markdown("#### Sources")
                        sources = gr.HTML(muted("Retrieved sources appear here."))

                inputs = [question, chatbot, top_k, threshold]
                outputs = [chatbot, question, metrics, sources]
                send.click(ask, inputs, outputs)
                question.submit(ask, inputs, outputs)
                clear.click(lambda: ([], answer_metrics_html(None), muted("Retrieved sources appear here.")),
                            None, [chatbot, metrics, sources])

            # ---------------- Retrieval inspector ----------------
            with gr.Tab("Retrieval Inspector"):
                gr.Markdown("See exactly which chunks reach the LLM. Useful for tuning Top-K and the relevance threshold.")
                with gr.Row():
                    probe = gr.Textbox(placeholder="Query", show_label=False, scale=5)
                    probe_btn = gr.Button("Search", variant="primary", scale=1)
                with gr.Row():
                    probe_k = gr.Slider(1, 20, value=10, step=1, label="Top-K")
                    probe_th = gr.Slider(0.0, 0.9, value=RELEVANCE_THRESHOLD, step=0.01, label="Relevance threshold")
                probe_summary = gr.HTML()
                probe_chart = gr.BarPlot(x="rank", y="score", color="passes threshold", y_lim=[0, 1],
                                         sort=None, height=260, show_label=False)
                probe_table = gr.Dataframe(wrap=True, show_label=False)
                for trigger in (probe_btn.click, probe.submit):
                    trigger(inspect_retrieval, [probe, probe_k, probe_th], [probe_table, probe_chart, probe_summary])

            # ---------------- Evaluation ----------------
            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "Runs `evaluation/dataset.json`: in-scope questions labelled with the pages that answer them, "
                    "plus out-of-scope questions the system must refuse. Retrieval-only takes about 30 s; "
                    "with generation it takes several minutes."
                )
                with gr.Row():
                    eval_k = gr.Slider(1, 15, value=TOP_K, step=1, label="Top-K")
                    eval_th = gr.Slider(0.0, 0.9, value=RELEVANCE_THRESHOLD, step=0.01, label="Relevance threshold")
                    eval_gen = gr.Checkbox(label="Also generate answers (citations, end-to-end latency)", value=False)
                eval_btn = gr.Button("Run evaluation", variant="primary")
                eval_meta = gr.HTML()
                eval_cards = gr.HTML()
                eval_chart = gr.ScatterPlot(
                    x="question #", y="top score", color="type", y_lim=[0, 1], height=280,
                    label="Top retrieval score per question. In-scope and out-of-scope should be separated by the threshold.",
                )
                eval_table = gr.Dataframe(wrap=True, show_label=False)
                eval_outputs = [eval_cards, eval_table, eval_chart, eval_meta]
                eval_btn.click(run_eval, [eval_k, eval_th, eval_gen], eval_outputs)

            # ---------------- Monitoring ----------------
            with gr.Tab("Monitoring"):
                gr.Markdown("Every question asked in the dashboard is logged to `logs/queries.jsonl`.")
                with gr.Row():
                    mon_refresh = gr.Button("Refresh", size="sm")
                    mon_clear = gr.Button("Clear log", size="sm", variant="stop")
                mon_cards = gr.HTML()
                mon_chart = gr.BarPlot(x="query #", y="seconds", color="stage", height=280,
                                       label="Latency per question by stage")
                mon_table = gr.Dataframe(wrap=True, show_label=False)
                mon_outputs = [mon_cards, mon_chart, mon_table]
                mon_refresh.click(monitoring_views, None, mon_outputs)
                mon_clear.click(clear_monitoring, None, mon_outputs)

            # ---------------- Knowledge base ----------------
            with gr.Tab("Knowledge Base"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("#### System health")
                        kb_health = gr.HTML()
                        kb_refresh = gr.Button("Re-check", size="sm")
                    with gr.Column(scale=3):
                        gr.Markdown("#### Documents")
                        kb_docs = gr.Dataframe(show_label=False)
                        kb_file = gr.File(label=f"Add a document ({', '.join(sorted(SUPPORTED_EXTENSIONS))})",
                                          file_types=sorted(SUPPORTED_EXTENSIONS), type="filepath")
                        with gr.Row():
                            kb_upload = gr.Button("Upload & index", variant="primary")
                            kb_sync = gr.Button("Re-sync folder")
                        with gr.Row():
                            kb_remove_name = gr.Textbox(placeholder="document name to remove from the index", show_label=False, scale=3)
                            kb_remove = gr.Button("Remove", variant="stop", scale=1)
                        kb_status = gr.HTML()
                kb_outputs = [kb_status, kb_health, kb_docs]
                kb_refresh.click(kb_views, None, [kb_health, kb_docs])
                kb_upload.click(upload_and_index, kb_file, kb_outputs)
                kb_sync.click(resync, None, kb_outputs)
                kb_remove.click(remove_document, kb_remove_name, kb_outputs)

        demo.load(lambda: eval_views(load_latest()), None, eval_outputs)
        demo.load(monitoring_views, None, mon_outputs)
        demo.load(kb_views, None, [kb_health, kb_docs])

    return demo


THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
