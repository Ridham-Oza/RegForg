import gradio as gr


def create_header():

    return gr.Markdown(
        """
# 🤖 Qwen 8B RAG Intelligence

### Ask questions about your documents using

**Qwen3 Embedding 8B + Pinecone + Ollama + Gradio**

Upload your documents to:

`data/documents/`

Then ask a question below.
"""
    )


def create_question_box():

    return gr.Textbox(
        label="Ask your document",
        placeholder=(
            "Example: What are the types of learning?"
        ),
        lines=4
    )


def create_button():

    return gr.Button(
        "🚀 Ask Question",
        variant="primary",
        size="lg"
    )


def create_answer_box():

    return gr.Markdown(
        label="Answer"
    )


def create_sources_box():

    return gr.Markdown(
        label="Sources"
    )


def create_metrics_box():

    return gr.Markdown(
        label="Performance"
    )