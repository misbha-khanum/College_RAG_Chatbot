import json
import os
import re
import shutil
import tempfile
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from vectorize_documents import (
    DATA_DIR,
    VECTOR_DB_DIR,
    get_embeddings,
    ingest_files,
    list_indexed_documents,
)

working_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(working_dir, ".env"))
if not os.environ.get("GROQ_API_KEY"):
    config_path = os.path.join(working_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            os.environ["GROQ_API_KEY"] = json.load(f).get("GROQ_API_KEY", "")

DSU_SYSTEM_PROMPT = """You are **Campus Care**, the official AI Student Support Assistant for **Dayananda Sagar University (DSU)** in Bengaluru.
Your responses must be **accurate, thorough, well-formatted** (using clear bullet points or tables when appropriate), and **strictly grounded in the retrieved DSU knowledge base and uploaded documents**.

Guidelines:
1. Answer directly and completely using the retrieved DSU context.
2. Highlight key details such as campus locations, program names, eligibility, fees, or contact numbers clearly.
3. Maintain a warm, professional, student-friendly tone.
"""

DSU_NEGATIVE_PROMPT = """
- Do **NOT** fabricate DSU fees, dates, seat counts, policies, or program details not present in the retrieved context.
- Do **NOT** rely on outside assumptions when DSU-specific data is missing.
- If a specific detail is not available in the knowledge base, politely state that and direct the student to **admissions@dsu.edu.in** or **080 46461800**.
"""

STARTER_PROMPTS = [
    "What campuses and helplines does DSU have?",
    "List all DSU schools and engineering programs",
    "Tell me about DSU placements and scholarships",
    "What hostel, hospital, and innovation lab facilities are at DSU?",
]

# Load embeddings once at startup
embeddings = get_embeddings()


def _get_vectorstore() -> Chroma:
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)


def create_chat_chain():
    vectorstore = _get_vectorstore()
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    prompt_template = f"""{DSU_SYSTEM_PROMPT}

{DSU_NEGATIVE_PROMPT}

Context:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"],
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=False,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _format_sources_footer(source_docs) -> str:
    if not source_docs:
        return ""
    seen = set()
    items = []
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown")
        url = doc.metadata.get("source_url", "")
        key = (src, url)
        if key in seen:
            continue
        seen.add(key)
        if url:
            items.append(f"`{src}` ([Official Link]({url}))")
        else:
            items.append(f"`{src}`")
    if not items:
        return ""
    return "\n\n---\n**Sources:** " + " · ".join(items)


def _render_header_html() -> str:
    doc_count = len(list_indexed_documents())
    return f"""
    <div class="dsu-banner">
        <div>
            <div class="dsu-title">🎓 Campus Care — Dayananda Sagar University (DSU)</div>
            <div class="dsu-subtitle">AI Student Support Assistant · Admissions, Academics, Placements &amp; Campus Life</div>
        </div>
        <div class="dsu-badge">📚 {doc_count} Indexed DSU Docs</div>
    </div>
    """


def _render_indexed_docs_md() -> str:
    docs = list_indexed_documents()
    if not docs:
        return "_No documents indexed yet._"
    return "\n".join(f"- `{name}`" for name in docs)


def respond(message: str, history: list, chain):
    if not message or not message.strip():
        return "", history, chain

    if chain is None:
        chain = create_chat_chain()

    result = chain.invoke({"question": message})
    answer = result["answer"] + _format_sources_footer(result.get("source_documents", []))

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history, chain


def use_starter(prompt_text: str, history: list, chain):
    _, updated_history, updated_chain = respond(prompt_text, history, chain)
    return updated_history, updated_chain


def upload_and_index(uploaded_files, chain):
    if not uploaded_files:
        return (
            "⚠️ Please select one or more `.pdf`, `.txt`, or `.md` files first.",
            _render_indexed_docs_md(),
            _render_header_html(),
            chain,
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    saved_paths = []
    for file_obj in uploaded_files:
        src_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
        filename = os.path.basename(src_path)
        dest_path = os.path.join(DATA_DIR, filename)
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            shutil.copy2(src_path, dest_path)
        saved_paths.append(dest_path)

    total_chunks, summary_lines = ingest_files(saved_paths, embeddings=embeddings)
    # Hot-reload chain with updated Chroma retriever
    new_chain = create_chat_chain()
    status_msg = f"✅ Indexed **{total_chunks}** new chunks:\n" + "\n".join(summary_lines)
    return status_msg, _render_indexed_docs_md(), _render_header_html(), new_chain


def clear_chat():
    return [], create_chat_chain(), None


def export_pdf(history: list):
    if not history:
        return gr.update(visible=False)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "Campus Care - Dayananda Sagar University (DSU)", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Transcript Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(6)

    for msg in history:
        role = str(msg.get("role", "user")).capitalize()
        text = remove_emojis(_extract_text(msg.get("content", "")))
        safe_text = text.encode("latin-1", "replace").decode("latin-1")
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, role, ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"dsu_campus_care_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    )
    pdf.output(tmp_path)
    return gr.update(value=tmp_path, visible=True)


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.dsu-banner {
    background: linear-gradient(135deg, #0B3D91 0%, #1E40AF 100%);
    color: #FFFFFF;
    padding: 20px 24px;
    border-radius: 14px;
    border-bottom: 4px solid #F59E0B;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 6px 18px rgba(11, 61, 145, 0.18);
    margin-bottom: 12px;
}
.dsu-title {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.dsu-subtitle {
    font-size: 0.92rem;
    color: #E2E8F0;
    margin-top: 4px;
}
.dsu-badge {
    background: #F59E0B;
    color: #0F172A;
    font-weight: 700;
    font-size: 0.88rem;
    padding: 8px 14px;
    border-radius: 999px;
    white-space: nowrap;
}
.dsu-card {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}
.dsu-starter-btn button {
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    background: #F8FAFC !important;
    color: #0B3D91 !important;
    text-align: left !important;
}
.dsu-starter-btn button:hover {
    border-color: #F59E0B !important;
    background: #FFFBEB !important;
}
"""

DSU_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#0B3D91",
    button_primary_background_fill_hover="#1E40AF",
    button_primary_text_color="#FFFFFF",
)


with gr.Blocks(title="Campus Care — Dayananda Sagar University (DSU)") as demo:
    chain_state = gr.State(None)

    header_html = gr.HTML(_render_header_html())

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["dsu-card"]):
                gr.Markdown(
                    """
### 🏛️ DSU Quick Contact & Campuses
- **Main Campus:** Devarakaggalahalli, Harohalli, Kanakapura Rd, Bengaluru – 562112
- **City Innovation Campus:** Hosur Main Rd, Kudlu Gate, Bengaluru – 560068
- **Admissions Helpline:** `080 46461800`
- **Email:** `admissions@dsu.edu.in`
- **Web:** [dsu.edu.in](https://www.dsu.edu.in/) · [admissions.dsu.edu.in](https://admissions.dsu.edu.in/)
"""
                )

            with gr.Group(elem_classes=["dsu-card"]):
                gr.Markdown("### 📂 Knowledge Base & Upload")
                file_uploader = gr.File(
                    label="Add DSU PDFs, TXT, or Markdown",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                )
                upload_btn = gr.Button("Upload & Index", variant="primary")
                upload_status = gr.Markdown("")
                with gr.Accordion("Indexed Documents", open=False):
                    indexed_docs_md = gr.Markdown(_render_indexed_docs_md())

            with gr.Group(elem_classes=["dsu-card"]):
                with gr.Row():
                    export_btn = gr.Button("📄 Export PDF")
                    clear_btn = gr.Button("🗑️ Clear Chat")
                pdf_file = gr.File(label="Download Transcript", visible=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="Campus Care Assistant",
                buttons=["copy", "copy_all"],
                placeholder="Ask anything about Dayananda Sagar University (DSU)...",
            )

            with gr.Row():
                starter_btns = []
                for prompt_text in STARTER_PROMPTS:
                    btn = gr.Button(
                        prompt_text,
                        size="sm",
                        elem_classes=["dsu-starter-btn"],
                    )
                    starter_btns.append((btn, prompt_text))

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about DSU admissions, DSAT, B.Tech/M.Sc/MBA programs, fees, hostels, or placements...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    msg_input.submit(
        respond,
        inputs=[msg_input, chatbot, chain_state],
        outputs=[msg_input, chatbot, chain_state],
    )
    send_btn.click(
        respond,
        inputs=[msg_input, chatbot, chain_state],
        outputs=[msg_input, chatbot, chain_state],
    )
    for btn, prompt_text in starter_btns:
        btn.click(
            use_starter,
            inputs=[gr.State(prompt_text), chatbot, chain_state],
            outputs=[chatbot, chain_state],
        )
    upload_btn.click(
        upload_and_index,
        inputs=[file_uploader, chain_state],
        outputs=[upload_status, indexed_docs_md, header_html, chain_state],
    )
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, chain_state, pdf_file],
    )
    export_btn.click(
        export_pdf,
        inputs=[chatbot],
        outputs=[pdf_file],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        inbrowser=True,
        theme=DSU_THEME,
        css=CUSTOM_CSS,
    )
