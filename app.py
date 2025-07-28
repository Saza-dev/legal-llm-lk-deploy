import streamlit as st
from streamlit_option_menu import option_menu

# --- SIDEBAR API Inputs ---
st.sidebar.title("🔐 API Keys")
hf_key = st.sidebar.text_input("Hugging Face API Key", type="password")
groq_key = st.sidebar.text_input("Groq API Key", type="password")

# Save in session
st.session_state["HUGGINGFACE_API_KEY"] = hf_key
st.session_state["GROQ_API_KEY"] = groq_key

# --- Main Menu ---
selected = option_menu(
    menu_title=None,
    options=["Assistant", "Drafter", "Compliance Checker", "Summarizer"],
    icons=["robot", "envelope", "check", "body-text"],
    orientation="horizontal"
)

# Validate keys before continuing
if not hf_key or not groq_key:
    st.warning("Please enter both API keys in the sidebar.")
else:
    # Now import dynamically based on selected section
    from chains import assistant_chain, drafter_chain, compliance_chain, summarizer_chain

    if selected == "Assistant":
        assistant_chain.run()
    elif selected == "Drafter":
        drafter_chain.run()
    elif selected == "Compliance Checker":
        compliance_chain.run()
    elif selected == "Summarizer":
        summarizer_chain.run()
