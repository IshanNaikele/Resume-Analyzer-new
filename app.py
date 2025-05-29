import streamlit as st
import pickle
import re
import nltk
from docx import Document  
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load pre-trained model and TF-IDF vectorizer
@st.cache_resource
def load_models():
    try:
        log_model = pickle.load(open('log_model.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        return log_model, tfidf
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

log_model, tfidf = load_models()

# Label mapping
label_map = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
    4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science',
    7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer',
    10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR',
    13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer',
    16: 'Mechanical Engineer', 17: 'Network Security Engineer',
    18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer',
    21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+', ' URL ', txt)
    txt = re.sub(r'rt|cc', ' ', txt)
    txt = re.sub(r'#\S+', ' HASHTAG ', txt)
    txt = re.sub(r'@\S+', ' MENTION ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    txt = ' '.join([lemmatizer.lemmatize(word) for word in txt.split() if word not in stop_words])
    return txt


# Improved text extraction functions
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

def extract_text_from_txt(file):
    try:
        # Read as bytes first
        file_bytes = file.read()
        
        # Try UTF-8 decoding first
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Try common fallback encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return file_bytes.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return file_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        st.error(f"TXT extraction error: {str(e)}")
        return ""

# Unified file handler
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""

# Prediction function optimized
def predict_category(resume_text):
    cleaned_text = cleanResume(resume_text)
    if not cleaned_text.strip():
        return "Text extraction failed - no content to analyze"
    
    try:
        vectorized_text = tfidf.transform([cleaned_text])
        predicted_label = log_model.predict(vectorized_text)[0]
        return label_map.get(predicted_label, "Unknown category")
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Streamlit UI with enhanced features
def main():
     
    
    st.title("📄 Resume Category Classifier")
    st.markdown("""
    <style>
    .big-font { font-size:24px !important; }
    .result-box { border: 2px solid #4CAF50; border-radius: 5px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("About")
        st.info("This app classifies resumes into 25 job categories using machine learning.")
        st.markdown("""
        - Upload PDF, DOCX, or TXT files
        - Models trained on 962 resumes
        - Cleaned and processed text
        """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF, DOCX, or TXT)", 
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 10MB"
    )
    
    if uploaded_file:
        with st.spinner("Processing your resume..."):
            # Process file
            resume_text = handle_file_upload(uploaded_file)
            
            if resume_text:
                # Display file info
                st.success(f"File uploaded: **{uploaded_file.name}**")
                
                # Show extracted text with toggle
                with st.expander("View extracted text", expanded=False):
                    st.text_area("Resume Content", resume_text, height=300)
                
                # Show cleaned text with toggle
                cleaned_text = cleanResume(resume_text)
                with st.expander("View cleaned text", expanded=False):
                    st.text_area("Cleaned Text", cleaned_text, height=300)
                
                # Make prediction
                category = predict_category(resume_text)
                
                # Display result with styling
                st.markdown("### Prediction Result")
                st.markdown(f'<div class="result-box"><p class="big-font">Predicted Category: <strong>{category}</strong></p></div>', 
                           unsafe_allow_html=True)
                
                # Show confidence score if available
                if hasattr(log_model, "predict_proba"):
                    vectorized_text = tfidf.transform([cleaned_text])
                    probas = log_model.predict_proba(vectorized_text)[0]
                    top_3 = sorted(zip(label_map.values(), probas), key=lambda x: x[1], reverse=True)[:3]
                    
                    st.subheader("Top 3 Predictions")
                    for cat, prob in top_3:
                        st.progress(prob, text=f"{cat}: {prob:.1%}")

if __name__ == "__main__":
    main()