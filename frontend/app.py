#app.py
import streamlit as st
from client import LLMClient


st.set_page_config(page_title="Chat RAG - Consejo Universitario", page_icon="📋", layout="centered")
st.title("Resoluciones LLM Chatbot")

client = LLMClient()

# History 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Escribe tu consulta sobre las resoluciones del Consejo Universitario..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call LLM client
    response = client.generate_text(prompt)

    # Preprocess the response
    if isinstance(response, dict) and "answer" in response:
        answer = response["answer"]
        citations = response.get("citations", [])

        full_response_markdown = answer

        with st.chat_message("assistant"):
            st.markdown(full_response_markdown)

            # Citations bloc
            if citations:
                with st.expander("Ver citas y referencias"):
                    for c in citations:
                        st.markdown(f"- **{c['id_reso']}** ({c['fecha']})")
                        st.write(f"  *{c['extracto']}*")
    else: 
        with st.chat_message("assistant"):
            st.markdown("Chat error... Por favor, intenta de nuevo más tarde.")
    
    if isinstance(response, dict) and "answer" in response:
        st.session_state.messages.append({"role": "assistant", "content": answer})

    

            