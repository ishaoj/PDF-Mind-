import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'processed_doc' not in st.session_state:
        st.session_state.processed_doc = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None

def display_chat_message(role, content):
    with st.chat_message(role):
        st.write(content)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto"  # Automatically collapse on mobile
    )

    # Custom CSS for better responsiveness
    st.markdown("""
        <style>
            .main > div {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .stButton>button {
                width: 100%;
            }
            .st-emotion-cache-16idsys p {
                word-break: break-word;
            }
            @media (max-width: 768px) {
                .main > div {
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        doc = st.file_uploader(
            "Upload your PDF document",
            type=['pdf'],
            help="Please upload a PDF file to begin asking questions",
            label_visibility="collapsed"
        )
        
        if doc:
            if st.session_state.processed_doc != doc:
                with st.spinner("Processing document..."):
                    try:
                        document = load_data(doc)
                        model = load_model()
                        st.session_state.query_engine = download_gemini_embedding(model, document)
                        st.session_state.processed_doc = doc
                        st.success(f"✅ {doc.name} processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        st.session_state.processed_doc = None
                        st.session_state.query_engine = None
        
        # Instructions - collapsible on mobile
        with st.expander("How to use", expanded=True):
            st.markdown("""
                1. Upload your PDF document
                2. Type your question
                3. Press Enter or click Send
                4. View responses in the chat
            """)

    # Main content area
    st.title("Document Q&A System")

    # Container for chat interface
    chat_container = st.container()
    
    # Question input - always at bottom
    user_question = st.chat_input(
        "Ask a question about your document",
        disabled=not st.session_state.query_engine
    )

    # Display chat history in reverse order (newest first)
    with chat_container:
        for q, a in reversed(st.session_state.history):
            display_chat_message("user", q)
            display_chat_message("assistant", a)

    # Process question
    if user_question and st.session_state.query_engine:
        display_chat_message("user", user_question)
        
        with st.spinner(""):
            try:
                response = st.session_state.query_engine.query(user_question)
                display_chat_message("assistant", response.response)
                
                # Add to history
                st.session_state.history.append((user_question, response.response))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Document upload prompt
    if not st.session_state.processed_doc:
        st.info("👈 Please upload a document to begin")

if __name__ == "__main__":
    main()