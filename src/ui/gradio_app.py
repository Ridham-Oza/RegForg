import html
import time
import gradio as gr

from src.pipeline.rag_pipeline import run_rag


# ============================================================
# REGFORG — RAG KNOWLEDGE ASSISTANT
# ============================================================

APP_TITLE = "RegForg — RAG Knowledge Assistant"


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = r"""
/* =========================================================
   GLOBAL
   ========================================================= */

:root {
    --bg: #0b0f19;
    --panel: #111827;
    --panel-2: #151d2e;
    --border: #263247;
    --text: #f3f4f6;
    --muted: #9ca3af;
    --accent: #f97316;
    --accent-hover: #ea580c;
    --green: #22c55e;
    --blue: #38bdf8;
}

body {
    background: var(--bg) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background:
        radial-gradient(
            circle at top right,
            rgba(249, 115, 22, 0.08),
            transparent 35%
        ),
        var(--bg) !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.hero {
    padding: 42px 30px 30px 30px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(249,115,22,0.12),
            rgba(56,189,248,0.05)
        ),
        var(--panel);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 17px;
    line-height: 1.6;
}

.tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.tech-badge {
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    color: #dbeafe;
    font-size: 13px;
    font-weight: 600;
}


/* =========================================================
   QUESTION AREA
   ========================================================= */

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

.question-box textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-size: 16px !important;
    padding: 16px !important;
}

.ask-btn {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    min-height: 52px !important;
}

.ask-btn:hover {
    background: var(--accent-hover) !important;
}

.clear-btn {
    border-radius: 14px !important;
    min-height: 52px !important;
}


/* =========================================================
   ANSWER
   ========================================================= */

.answer-panel {
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    background: var(--panel);
    min-height: 160px;
}

.answer-label {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 14px;
}

.answer-text {
    color: #e5e7eb;
    font-size: 16px;
    line-height: 1.8;
}


/* =========================================================
   METRICS
   ========================================================= */

.metric-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--panel);
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: white;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.metric-accent {
    color: var(--accent);
}

.metric-green {
    color: var(--green);
}

.metric-blue {
    color: var(--blue);
}


/* =========================================================
   SOURCES
   ========================================================= */

.source-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--panel);
    padding: 20px;
    margin-bottom: 12px;
}

.source-header {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    align-items: center;
    margin-bottom: 12px;
}

.source-title {
    font-weight: 800;
    color: #f9fafb;
}

.source-score {
    color: var(--green);
    font-weight: 700;
    font-size: 13px;
}

.source-meta {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 12px;
}

.source-text {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.65;
    background: rgba(255,255,255,0.025);
    border-radius: 10px;
    padding: 13px;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 10px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
    padding: 30px 10px;
}

.footer strong {
    color: #9ca3af;
}
"""


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    """Safely escape text before putting it into HTML."""
    if value is None:
        return ""

    return html.escape(str(value))


def format_seconds(value):
    """Format seconds for UI."""
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "—"


def build_answer_html(answer):
    """Create polished answer panel."""

    if not answer:
        return """
        <div class="answer-panel">
            <div class="answer-label">💡 Answer</div>
            <div class="answer-text">
                Ask a question about the Machine Learning book.
            </div>
        </div>
        """

    formatted = safe_text(answer).replace("\n", "<br>")

    return f"""
    <div class="answer-panel">
        <div class="answer-label">💡 Answer</div>
        <div class="answer-text">
            {formatted}
        </div>
    </div>
    """


def build_metrics_html(result):
    """Create metrics dashboard."""

    if not result:
        return """
        <div class="status">
            Retrieval and generation metrics will appear here.
        </div>
        """

    retrieval_time = result.get("retrieval_time", 0)
    generation_time = result.get("generation_time", 0)
    total_time = result.get("total_time", 0)
    num_chunks = result.get("num_chunks", 0)

    return f"""
    <div style="margin-top:18px;">
        <div class="section-title">⚡ RAG Performance</div>

        <div style="
            display:grid;
            grid-template-columns:
                repeat(auto-fit, minmax(170px, 1fr));
            gap:12px;
        ">

            <div class="metric-card">
                <div class="metric-value metric-blue">
                    {format_seconds(retrieval_time)}
                </div>
                <div class="metric-label">
                    RETRIEVAL TIME
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-value metric-accent">
                    {format_seconds(generation_time)}
                </div>
                <div class="metric-label">
                    GENERATION TIME
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-value metric-green">
                    {format_seconds(total_time)}
                </div>
                <div class="metric-label">
                    TOTAL TIME
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-value">
                    {num_chunks}
                </div>
                <div class="metric-label">
                    CHUNKS USED
                </div>
            </div>

        </div>
    </div>
    """


def build_sources_html(result):
    """Create source cards."""

    if not result:
        return """
        <div class="status">
            Retrieved source documents will appear here.
        </div>
        """

    documents = result.get("documents", [])

    if not documents:
        return """
        <div class="status">
            No source documents were retrieved.
        </div>
        """

    cards = []

    for i, document in enumerate(documents, start=1):

        file_name = safe_text(
            document.get("file_name", "Unknown")
        )

        page = safe_text(
            document.get("page", "Unknown")
        )

        score = document.get("score", 0)

        text = safe_text(
            document.get("text", "")
        )

        # Keep UI readable.
        if len(text) > 1800:
            text = text[:1800] + "..."

        cards.append(
            f"""
            <div class="source-card">

                <div class="source-header">

                    <div class="source-title">
                        📄 Source {i}
                    </div>

                    <div class="source-score">
                        {float(score):.4f}
                    </div>

                </div>

                <div class="source-meta">
                    <strong>File:</strong> {file_name}
                    &nbsp;&nbsp;•&nbsp;&nbsp;
                    <strong>Page:</strong> {page}
                </div>

                <div class="source-text">
                    {text}
                </div>

            </div>
            """
        )

    return "".join(cards)


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_question(question):
    """
    Main UI function.

    Calls the existing RAG pipeline.
    """

    question = (question or "").strip()

    if not question:
        return (
            build_answer_html(
                "Please enter a question."
            ),
            build_metrics_html(None),
            build_sources_html(None),
            "⚠️ Please enter a question."
        )

    try:

        start = time.perf_counter()

        result = run_rag(
            question,
            top_k=5
        )

        elapsed = time.perf_counter() - start

        answer = result.get(
            "answer",
            "No answer generated."
        )

        status = (
            f"✅ Answer generated successfully "
            f"in {elapsed:.2f}s"
        )

        return (
            build_answer_html(answer),
            build_metrics_html(result),
            build_sources_html(result),
            status
        )

    except Exception as error:

        error_message = safe_text(str(error))

        answer_html = f"""
        <div class="answer-panel">
            <div class="answer-label">
                ❌ Something went wrong
            </div>

            <div class="answer-text">
                {error_message}
            </div>
        </div>
        """

        return (
            answer_html,
            build_metrics_html(None),
            build_sources_html(None),
            "❌ RAG request failed."
        )


# ============================================================
# CLEAR
# ============================================================

def clear_all():
    return (
        "",
        build_answer_html(None),
        build_metrics_html(None),
        build_sources_html(None),
        "Ready."
    )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    
) as demo:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-title">
                📚 RegForg
            </div>

            <div class="hero-subtitle">
                RAG Knowledge Assistant for the
                Hundred-Page Machine Learning Book
            </div>

            <div class="tech-row">

                <div class="tech-badge">
                    Qwen3 Embedding 8B
                </div>

                <div class="tech-badge">
                    4096 Dimensions
                </div>

                <div class="tech-badge">
                    Pinecone
                </div>

                <div class="tech-badge">
                    Qwen3 8B
                </div>

                <div class="tech-badge">
                    Retrieval-Augmented Generation
                </div>

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    gr.Markdown(
        "### 🔎 Ask the Knowledge Base"
    )

    with gr.Row():

        with gr.Column(
            scale=5
        ):

            question = gr.Textbox(
                label="Your Question",
                placeholder=(
                    "Example: What is supervised learning?"
                ),
                lines=4,
                max_lines=8,
                elem_classes=["question-box"]
            )

        with gr.Column(
            scale=1,
            min_width=170
        ):

            ask_button = gr.Button(
                "🔍 Ask",
                variant="primary",
                elem_classes=["ask-btn"]
            )

            clear_button = gr.Button(
                "🗑 Clear",
                variant="secondary",
                elem_classes=["clear-btn"]
            )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    gr.Markdown(
        "### 💬 Try an example"
    )

    gr.Examples(
        examples=[
            "What is machine learning?",
            "What is supervised learning?",
            "What is unsupervised learning?",
            "Explain reinforcement learning.",
            "What is a feature vector?",
            "What is overfitting?",
        ],
        inputs=question
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = gr.Markdown(
        "Ready.",
        elem_classes=["status"]
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    answer_output = gr.HTML(
        build_answer_html(None)
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics_output = gr.HTML(
        build_metrics_html(None)
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    gr.Markdown(
        "### 📚 Retrieved Sources"
    )

    sources_output = gr.HTML(
        build_sources_html(None)
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">
            <strong>RegForg</strong>
            &nbsp;•&nbsp;
            Qwen3 Embedding 8B
            &nbsp;•&nbsp;
            Pinecone
            &nbsp;•&nbsp;
            Qwen3 8B
            &nbsp;•&nbsp;
            Local RAG
        </div>
        """
    )

    # ========================================================
    # EVENTS
    # ========================================================

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer_output,
            metrics_output,
            sources_output,
            status
        ]
    )

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer_output,
            metrics_output,
            sources_output,
            status
        ]
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            question,
            answer_output,
            metrics_output,
            sources_output,
            status
        ]
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("REGFORG — RAG KNOWLEDGE ASSISTANT")
    print("=" * 70)
    print()
    print("Starting Gradio server...")
    print("Open the URL shown below in your browser.")
    print("=" * 70)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        css=CSS,
        show_error=True
    )
