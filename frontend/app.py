#app.py
import streamlit as st
from client import LLMClient
from streamlit_lottie import st_lottie
import json
import time
from uuid import uuid4
from datetime import datetime


def write_progresive_message(message: str, delay: float = 0.02):
    """Writes a message progressively in the Streamlit app."""
    placeholder = st.empty()
    buffer = ""
    for char in message:
        buffer += char
        placeholder.markdown(buffer)
        time.sleep(delay)

def send_feedback(reaction: str, trace_id: str):
    fb_payload = {
        "trace_id": trace_id,
        "session_id": st.session_state.session_id,
        "prompt": st.session_state.last_prompt,
        "feedback": reaction,
        "backend_response": st.session_state.last_response,
        "extra": {
            "ui_ts": datetime.now(datetime.astimezone.utc).isoformat(),
            "component": "streamlit",
            "version": st.__version__

        }
    }
    ok, _ = client.send_feedback(fb_payload)
    if ok:
        st.session_state.feedback_sent[trace_id] = reaction
        st.toast("¡Gracias por tu feedback!")
    else:
        st.warning("No se pudo enviar el feedback. Por favor, intenta de nuevo más tarde.")
    

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

# History 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Escribe tu consulta sobre las resoluciones del Consejo Universitario...", ):
    st.session_state.messages.append({"role": "user", "content": prompt})
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
        trace_id = response.get("trace_id")

        write_progresive_message(answer)

        # Citations bloc
        if citations:
            with st.expander("Ver citas y referencias"):
                for c in citations:
                    st.markdown(f"- **{c.get('id_reso','')}** ({c.get('fecha','')})")
                    st.write(f"  *{c.get('extracto','')}*")

        # Feedback buttons
        already = st.session_state.feedback_sent.get(trace_id)
        col1, col2 = st.columns(2)
        with col1:
            like_disabled = already is not None
            if st.button("Útil", key=f"like_{trace_id}", icon=":material/thumb_up:", disabled=like_disabled):
                send_feedback("like", trace_id)
                
        with col2:
            if st.button("No útil", key=f"dislike_{len(st.session_state.messages)}", icon=":material/thumb_down:"):
                send_feedback("dislike", trace_id)
        st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
    else:
        st.markdown("Chat error... Por favor, intenta de nuevo más tarde.")
    

            