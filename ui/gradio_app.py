import gradio as gr

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    TOP_K
)

from src.pipeline.rag_pipeline import (
    run_rag
)


def format_sources(
    documents
):

    if not documents:

        return (
            "### Sources\n\n"
            "No sources found."
        )

    output = [
        "### 📚 Sources"
    ]

    for i, document in enumerate(
        documents,
        start=1
    ):

        score = document["score"]

        output.append(
            f"""
**{i}. {document["file_name"]}**

- Page: `{document["page"]}`
- Relevance: `{score:.4f}`

> {document["text"][:600]}...
"""
        )

    return "\n".join(
        output
    )


def format_performance(
    result
):

    return f"""
### 📊 Performance

| Metric | Value |
|---|---:|
| Retrieval Time | `{result["retrieval_time"]:.3f}s` |
| Generation Time | `{result["generation_time"]:.3f}s` |
| Total Answer Time | `{result["total_time"]:.3f}s` |
| Retrieved Chunks | `{result["num_chunks"]}` |
| Embedding Model | `{EMBEDDING_MODEL}` |
| Embedding Dimension | `{PINECONE_DIMENSION}` |
| Vector Database | `Pinecone` |
| LLM | `{LLM_MODEL}` |
"""


def answer_question(
    question
):

    if not question or not question.strip():

        return (
            "### 🤖 Answer\n\n"
            "Please enter a question.",
            "### 📚 Sources\n\n"
            "No question provided.",
            "### 📊 Performance\n\n"
            "Waiting for a question."
        )

    try:

        result = run_rag(
            question=question.strip(),
            top_k=TOP_K
        )

        answer = result["answer"]

        sources = format_sources(
            result["documents"]
        )

        performance = format_performance(
            result
        )

        return (
            f"### 🤖 Answer\n\n{answer}",
            sources,
            performance
        )

    except Exception as e:

        return (
            "### ❌ Error\n\n"
            f"`{str(e)}`",
            "### 📚 Sources\n\n"
            "No sources available.",
            "### 📊 Performance\n\n"
            "Pipeline failed."
        )


def create_gradio_app():

    with gr.Blocks(
        title="Qwen 8B RAG Q&A"
    ) as demo:

        gr.Markdown(
            """
# 🤖 Qwen 8B RAG Q&A

### Local AI Document Intelligence

**Qwen3 Embedding 8B**
→ **Pinecone**
→ **Ollama**
→ **Qwen LLM**
→ **Gradio**
"""
        )

        with gr.Row():

            with gr.Column(
                scale=2
            ):

                question = gr.Textbox(
                    label="Ask your document",
                    placeholder=(
                        "What are the types of learning?"
                    ),
                    lines=5
                )

                ask_button = gr.Button(
                    "🚀 Ask Question",
                    variant="primary",
                    size="lg"
                )

                clear_button = gr.Button(
                    "🧹 Clear"
                )

            with gr.Column(
                scale=4
            ):

                answer = gr.Markdown(
                    """
### 🤖 Answer

Your answer will appear here.
"""
                )

        with gr.Row():

            with gr.Column():

                sources = gr.Markdown(
                    """
### 📚 Sources

Sources will appear here.
"""
                )

            with gr.Column():

                performance = gr.Markdown(
                    """
### 📊 Performance

Metrics will appear here.
"""
                )

        ask_button.click(
            fn=answer_question,
            inputs=question,
            outputs=[
                answer,
                sources,
                performance
            ]
        )

        question.submit(
            fn=answer_question,
            inputs=question,
            outputs=[
                answer,
                sources,
                performance
            ]
        )

        clear_button.click(
            fn=lambda: (
                "",
                "### 🤖 Answer\n\n",
                "### 📚 Sources\n\n",
                "### 📊 Performance\n\n"
            ),
            inputs=None,
            outputs=[
                question,
                answer,
                sources,
                performance
            ]
        )

        gr.Markdown(
            f"""
---

### ⚙️ System

**Embedding:** `{EMBEDDING_MODEL}`

**Vector Database:** `Pinecone`

**Index:** `{PINECONE_INDEX_NAME}`

**Dimension:** `{PINECONE_DIMENSION}`

**LLM:** `{LLM_MODEL}`
"""
        )

    return demo