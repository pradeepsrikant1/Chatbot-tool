import PyPDF2
import io

def process_pdf(pdf_file):
    """
    Extract text from uploaded PDF file
    """
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text 