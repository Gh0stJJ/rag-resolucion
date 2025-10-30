#app.py
import streamlit as st
from client import LLMClient
from streamlit_lottie import st_lottie
import json
import time
from uuid import uuid4
from datetime import datetime, timezone


def write_progresive_message(message: str, delay: float = 0.01):
    """Writes a message progressively in the Streamlit app."""
    placeholder = st.empty()
    buffer = ""
    for char in message:
        buffer += char
        placeholder.markdown(buffer)
        time.sleep(delay)

def send_feedback(reaction: str, trace_id: str, backend_response: dict, comment: str | None = None):
    fb_payload = {
        "trace_id": trace_id,
        "session_id": st.session_state.session_id,
        "prompt": st.session_state.last_prompt,
        "feedback": reaction,
        "backend_response": backend_response,
        "extra": {
            "ui_ts": datetime.now(timezone.utc).isoformat(),
            "component": "streamlit",
            "version": st.__version__,
            "comment": comment

        }
    }
    ok, _ = client.send_feedback(fb_payload)
    if ok:
        st.session_state.feedback_sent[trace_id] = reaction
        st.toast("¡Gracias por tu feedback!")
    else:
        st.warning("No se pudo enviar el feedback. Por favor, intenta de nuevo más tarde.")

    st.rerun()

def render_feedback_controls(trace_id: str, backend_response: dict):
    """Render feedback buttons for a given trace_id."""
    already = st.session_state.feedback_sent.get(trace_id)
    pending = st.session_state.get("pending_feedback")

    if pending and pending.get("trace_id") == trace_id and not already:
        with st.form(f"fb_form_{trace_id}"):
            comment = st.text_area(
                "Cuéntanos qué faltó o qué estuvo mal (opcional)",
                max_chars=500,
                placeholder="Ej.: La respuesta no citó la resolución correcta; Faltaron detalles de la sección RESUELVE; etc."
            )
            colA, colB = st.columns(2)
            send_with_comment = colA.form_submit_button("Enviar comentario y enviar feedback")
            send_without_comment = colB.form_submit_button("Enviar sin comentario")
        if send_with_comment:
            st.session_state.pending_feedback = None
            send_feedback("dislike", trace_id, backend_response=backend_response, comment=comment or None)
        if send_without_comment:
            st.session_state.pending_feedback = None
            send_feedback("dislike", trace_id, backend_response=backend_response, comment=None)
        return
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Útil", key=f"like_{trace_id}", disabled=(already is not None)):
            send_feedback("like", trace_id, backend_response=backend_response)
    with col2:
        if st.button("No útil", key=f"dislike_{trace_id}", disabled=(already is not None)):
            st.session_state.pending_feedback = {"trace_id": trace_id}
            st.rerun()

def render_assistant_block(msg: dict):
    """Draw an assistant message block. (message, avatar, citations, feedback)"""
    with st.chat_message("assistant", avatar="assets/cutinbot.png"):
        st.markdown(msg["content"])
        citations = msg.get("citations") or []
        if citations:
            with st.expander("Ver citas y referencias"):
                for c in citations:
                    st.markdown(f"- **{c.get('id_reso','')}** ({c.get('fecha','')})")
                    st.write(f"  *{c.get('extracto','')}*")

        trace_id = msg.get("trace_id")
        backend_response = msg.get("backend_response") or st.session_state.last_response
        if trace_id:
            render_feedback_controls(trace_id, backend_response)
    
# Initial setup
st.set_page_config(page_title="Chat RAG - Consejo Universitario", page_icon="📋", layout="centered")
st.title("Resoluciones LLM Chatbot")

client = LLMClient()


# Base state

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = {} # dict[trace_id] = "like" | "dislike"
if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None

# History 
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        render_assistant_block(msg)
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# User input
if prompt := st.chat_input("Escribe tu consulta sobre las resoluciones del Consejo Universitario...", ):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_prompt = prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="assets/cutinbot.png"):
        loading_placeholder = st.empty()
        dots_placeholder = st.empty()

        try:
            with open("assets/Searching.json", "r") as f:
                lottie_json = json.load(f)
            with loading_placeholder.container():
                st_lottie(lottie_json, speed=1, loop=True, quality="high", height=150, key="loading")
                # create animated dots in a loop
                for i in range(10):
                    dots_placeholder.markdown("Buscando resoluciones" + "." * (i % 4))
                    time.sleep(0.5)
        except Exception as e:
            loading_placeholder.warning(f"No se pudo cargar la animación de carga. Error: {e}")

    # Call LLM client
    response = client.generate_text(prompt)

    loading_placeholder.empty()
    dots_placeholder.empty()

    # Preprocess the response
    if isinstance(response, dict) and "answer" in response:
        st.session_state.last_response = response

        answer = response["answer"]
        citations = response.get("citations", [])
        # trace_id = response.get("trace_id")

        # Dummy trace_id for now
        trace_id = response.get("trace_id", str(uuid4()))

        with st.chat_message("assistant", avatar="assets/cutinbot.png"):
            write_progresive_message(answer)
            
            if citations:
                with st.expander("Ver citas y referencias"):
                    for c in citations:
                        st.markdown(f"- **{c.get('id_reso','')}** ({c.get('fecha','')})")
                        st.write(f"  *{c.get('extracto','')}*")
            
            if trace_id:
                render_feedback_controls(trace_id, backend_response=response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "trace_id": trace_id,
            "backend_response": response
        })
    else:
        st.error("Hubo un error al obtener la respuesta del servidor. Por favor, intenta de nuevo más tarde.")
    

            