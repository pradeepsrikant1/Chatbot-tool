import streamlit as st
from pdf_processor import process_pdf
from llm_model import SimpleQA
import traceback

def main():
    st.set_page_config(
        page_title="PDF Question Answering Bot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("PDF Question Answering Bot")
    
    try:
        # File upload
        uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
        
        if uploaded_file is not None:
            # Process PDF
            try:
                text_content = process_pdf(uploaded_file)
                st.success("PDF processed successfully!")
                
                # Question input
                question = st.text_input("Ask a question about your document:")
                
                if question:
                    # Initialize QA model
                    try:
                        qa_model = SimpleQA()
                        
                        # Get answer
                        with st.spinner("Thinking..."):
                            answer = qa_model.get_answer(question, text_content)
                            st.write("Answer:", answer)
                    except Exception as e:
                        st.error(f"Error in model processing: {str(e)}")
                        st.error("Please try with a different question or document")
            
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.error("Please make sure the PDF is readable and try again")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again")

if __name__ == "__main__":
    main() 