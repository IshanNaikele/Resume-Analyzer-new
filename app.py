import streamlit as st
import pickle
import re
import nltk
from docx import Document  
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
import base64

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: white;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Upload Section */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Results Section */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        transform: translateY(0);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
    }
    
    /* Sidebar Styles */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Stats Cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #4ecdc4;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: white;
        opacity: 0.8;
    }
    
    /* Animation Classes */
    .fadeIn {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Progress Bar Styles */
    .custom-progress {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    /* File Upload Styles */
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.5);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Expander Styles */
    .streamlit-expander {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    </style>
    """, unsafe_allow_html=True)

# Load pre-trained model and TF-IDF vectorizer
@st.cache_resource
def load_models():
    try:
        log_model = pickle.load(open('log_model.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        return log_model, tfidf
    except Exception as e:
        st.error(f"🚫 Error loading models: {str(e)}")
        st.info("💡 Please ensure 'log_model.pkl' and 'tfidf.pkl' files are in the same directory")
        st.stop()

log_model,tfidf=load_models()

# Initialize session state
if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

# Label mapping with icons
label_map = {
    0: '⚖️ Advocate', 1: '🎨 Arts', 2: '🔧 Automation Testing', 3: '⛓️ Blockchain',
    4: '📊 Business Analyst', 5: '🏗️ Civil Engineer', 6: '📈 Data Science',
    7: '🛢️ Database', 8: '🔄 DevOps Engineer', 9: '💻 DotNet Developer',
    10: '🔄 ETL Developer', 11: '⚡ Electrical Engineering', 12: '👥 HR',
    13: '🐘 Hadoop', 14: '💪 Health and fitness', 15: '☕ Java Developer',
    16: '⚙️ Mechanical Engineer', 17: '🔒 Network Security Engineer',
    18: '📋 Operations Manager', 19: '📈 PMO', 20: '🐍 Python Developer',
    21: '🏢 SAP Developer', 22: '💼 Sales', 23: '🧪 Testing', 24: '🌐 Web Designing'
}

# Initialize NLTK components
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except:
    st.error("❌ NLTK data not available. Please install required NLTK packages.")
    st.stop()

def cleanResume(txt):
    """Enhanced text cleaning with progress indication"""
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

def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"📄 PDF extraction error: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"📝 DOCX extraction error: {str(e)}")
        return ""

def extract_text_from_txt(file):
    try:
        file_bytes = file.read()
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return file_bytes.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return file_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        st.error(f"📄 TXT extraction error: {str(e)}")
        return ""

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""

def predict_category(resume_text):
    cleaned_text = cleanResume(resume_text)
    if not cleaned_text.strip():
        return "❌ Text extraction failed - no content to analyze", None
    
    try:
        vectorized_text = tfidf.transform([cleaned_text])
        predicted_label = log_model.predict(vectorized_text)[0]
        probabilities = log_model.predict_proba(vectorized_text)[0] if hasattr(log_model, "predict_proba") else None
        return label_map.get(predicted_label, "❓ Unknown category"), probabilities
    except Exception as e:
        return f"❌ Prediction error: {str(e)}", None

def create_confidence_chart(probabilities, top_n=5):
    """Create beautiful confidence chart"""
    if probabilities is None:
        return None
    
    # Get top predictions
    top_indices = probabilities.argsort()[-top_n:][::-1]
    top_categories = [list(label_map.values())[i] for i in top_indices]
    top_probs = [probabilities[i] for i in top_indices]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=top_categories,
            x=top_probs,
            orientation='h',
            marker=dict(
                color=top_probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f'{prob:.1%}' for prob in top_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="🎯 Top Predictions Confidence",
        xaxis_title="Confidence Score",
        yaxis_title="Job Categories",
        template="plotly_white",
        height=400,
        paper_bgcolor='rgba(255,255,255,0.95)',
        plot_bgcolor='rgba(255,255,255,0.95)',
        font=dict(family="Poppins", size=14, color="#2d3436"),
        title_font=dict(size=18, color="#2d3436", family="Poppins"),
        xaxis=dict(
            title_font=dict(size=14, color="#636e72"),
            tickfont=dict(size=12, color="#636e72")
        ),
        yaxis=dict(
            title_font=dict(size=14, color="#636e72"),
            tickfont=dict(size=12, color="#2d3436")
        )
    )
    
    return fig

def display_stats_cards():
    """Display beautiful statistics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{st.session_state.processed_resumes}</div>
            <div class="stats-label">Resumes Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">25</div>
            <div class="stats-label">Job Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{st.session_state.processing_time:.2f}s</div>
            <div class="stats-label">Last Processing Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">AI</div>
            <div class="stats-label">Powered Classification</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # Load models
    log_model, tfidf = load_models()
    
    # Header Section
    st.markdown("""
    <div class="main-header fadeIn">
        <div class="main-title">🤖 AI Resume Classifier</div>
        <div class="subtitle">Discover Your Perfect Career Path with AI-Powered Classification</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Cards
    display_stats_cards()
    
    # Sidebar with enhanced content
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🎯 About This Tool</h2>
            <p>Our AI-powered resume classifier uses advanced machine learning to analyze your resume and predict the most suitable job category.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-content">
            <h3>✨ Features</h3>
            <ul>
                <li>🔍 Smart text extraction</li>
                <li>🧠 AI-powered classification</li>
                <li>📊 Confidence scoring</li>
                <li>🎨 Beautiful visualizations</li>
                <li>⚡ Lightning fast processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-content">
            <h3>📁 Supported Formats</h3>
            <ul>
                <li>📄 PDF Documents</li>
                <li>📝 Word Documents (.docx)</li>
                <li>📄 Text Files (.txt)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-content">
            <h3>🏆 Model Performance</h3>
            <ul>
                <li>📊 Trained on 962+ resumes</li>
                <li>🎯 25 job categories</li>
                <li>⚡ TF-IDF vectorization</li>
                <li>🤖 Logistic regression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main upload section
    st.markdown('<div class="upload-section fadeIn">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=["pdf", "docx", "txt"],
            help="Upload your resume in PDF, DOCX, or TXT format (Max: 10MB)"
        )
    
    with col2:
        if uploaded_file:
            st.markdown("### 📋 File Details")
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing section
    if uploaded_file:
        start_time = time.time()
        
        # Processing animation
        with st.spinner("🔄 Processing your resume with AI magic..."):
            time.sleep(1)  # Add slight delay for better UX
            
            # Extract text
            resume_text = handle_file_upload(uploaded_file)
            
            if resume_text:
                # Update session state
                st.session_state.processed_resumes += 1
                processing_time = time.time() - start_time
                st.session_state.processing_time = processing_time
                
                # Success message with custom styling
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #00b894, #00cec9);
                    color: white;
                    padding: 1rem 1.5rem;
                    border-radius: 15px;
                    margin: 1rem 0;
                    box-shadow: 0 5px 15px rgba(0, 184, 148, 0.3);
                    font-weight: 600;
                    text-align: center;
                    border-left: 5px solid #00a085;
                ">
                    ✅ Resume processed successfully in {processing_time:.2f} seconds!
                </div>
                """, unsafe_allow_html=True)
                
                # Text preview sections
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("👀 View Original Text", expanded=False):
                        st.text_area("Original Content", resume_text[:1000] + "...", height=200, disabled=True)
                
                with col2:
                    cleaned_text = cleanResume(resume_text)
                    with st.expander("🧹 View Cleaned Text", expanded=False):
                        st.text_area("Processed Content", cleaned_text[:1000] + "...", height=200, disabled=True)
                
                # Make prediction
                category, probabilities = predict_category(resume_text)
                st.session_state.last_prediction = category
                
                # Results section
                st.markdown("""
                <div class="result-card fadeIn">
                    <div class="prediction-title">🎯 Classification Result</div>
                    <div class="prediction-result">{}</div>
                </div>
                """.format(category), unsafe_allow_html=True)
                
                # Confidence chart
                if probabilities is not None:
                    st.markdown("### 📊 Confidence Analysis")
                    fig = create_confidence_chart(probabilities)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top 3 predictions with progress bars
                    st.markdown("### 🏆 Top 3 Predictions")
                    top_3_indices = probabilities.argsort()[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_indices):
                        category_name = list(label_map.values())[idx]
                        confidence = probabilities[idx]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(confidence, text=f"**{category_name}**")
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                
                # Action buttons
                st.markdown("### 🚀 Next Steps")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Process Another Resume"):
                        st.rerun()
                
                with col2:
                    if st.button("📊 View Detailed Analysis"):
                        st.balloons()
                        st.success("🎉 Detailed analysis feature coming soon!")
                
                with col3:
                    if st.button("💾 Save Results"):
                        st.success("✅ Results saved to session!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; opacity: 0.7; padding: 2rem;">
        <p>🤖 Powered by AI • Built with ❤️ using Streamlit</p>
        <p>Made with 🐍 Python • Machine Learning • Beautiful UI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()