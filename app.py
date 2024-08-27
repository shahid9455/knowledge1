import streamlit as st
from st_audiorec import st_audiorec
import requests
import json
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from fpdf import FPDF

# ----------------------- IBM Watson Configurations -----------------------

# NLU Configuration
nlu_api_key = 'IHbYzsY18Sl7i3Wr-_9YrYjpARDKZRnkO2ETjR5mfvnP'
nlu_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/3f07153d-defe-42a0-8215-b0d2d480d44f'

nlu_authenticator = IAMAuthenticator(nlu_api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=nlu_authenticator
)
nlu.set_service_url(nlu_url)

# Speech to Text Configuration
stt_api_key = 'YOUR_IBM_WATSON_SPEECH_TO_TEXT_API_KEY'
stt_url = 'YOUR_IBM_WATSON_SPEECH_TO_TEXT_URL'

stt_authenticator = IAMAuthenticator(stt_api_key)
speech_to_text = SpeechToTextV1(authenticator=stt_authenticator)
speech_to_text.set_service_url(stt_url)

# ----------------------- AIML API Configuration -----------------------

class AIMLClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def chat_completions_create(self, model, messages):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": messages}
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()

aiml_client = AIMLClient(
    api_key="45228194012549f09d70dd18da5ff8a8",
    base_url="https://api.aimlapi.com"
)

# ----------------------- File Configuration -----------------------

filename = 'text_storage_with_keywords.txt'

# ----------------------- Streamlit Page Configuration -----------------------

st.set_page_config(page_title="KnowledgeBridge", layout="wide")

# ----------------------- Custom CSS Styles -----------------------

st.markdown("""
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111 !important;
        }
        /* Main Content */
        .main {
            background-color: #222 !important;
            color: #fff !important;
        }
        /* Text Areas and Inputs */
        .stTextArea, .stTextInput {
            background-color: #333 !important;
            color: #fff !important;
        }
        /* Buttons */
        .stButton>button {
            background-color: #444 !important;
            color: #fff !important;
            border: none;
            border-radius: 5px;
            padding: 0.5em 1em;
        }
        /* Headers */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #4CAF50 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Session State Initialization -----------------------

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ----------------------- PDF Generation Function -----------------------

def generate_pdf(texts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for text in texts:
        pdf.multi_cell(0, 10, text)
        pdf.ln(5)
    return pdf

# ----------------------- Navigation Sidebar -----------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input", "Search"])

# ----------------------- Input Page -----------------------

if page == "Input":
    st.title("📥 Input Page")

    # ----------------------- Audio Recording Section -----------------------

    st.header("🎤 Voice Recording")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        with st.spinner("Transcribing audio..."):
            try:
                audio_stream = BytesIO(wav_audio_data)
                recognition_result = speech_to_text.recognize(
                    audio=audio_stream,
                    content_type='audio/wav',
                    model='en-US_BroadbandModel',
                    max_alternatives=1
                ).get_result()
                
                transcript = recognition_result['results'][0]['alternatives'][0]['transcript']
                st.session_state.transcribed_text = transcript
                st.success("Audio transcribed successfully!")
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")

    # ----------------------- Text Input Section -----------------------

    st.header("📝 Text Input")
    text_input = st.text_area("Enter text to save:", value=st.session_state.transcribed_text, height=200)

    if st.button("Save Text"):
        if text_input.strip():
            with st.spinner("Extracting keywords and saving text..."):
                try:
                    response = nlu.analyze(
                        text=text_input,
                        features=Features(
                            keywords=KeywordsOptions(limit=15)
                        )
                    ).get_result()

                    keywords = [kw['text'] for kw in response.get('keywords', [])]
                    keyword_string = ', '.join(keywords)

                    with open(filename, 'a') as file:
                        file.write(f"{text_input}\nKeywords: {keyword_string}\n\n")
                    
                    st.session_state.uploaded_text = text_input
                    st.success("Text and keywords saved successfully!")
                except Exception as e:
                    st.error(f"Error extracting keywords: {e}")
        else:
            st.warning("Please enter some text to save.")

    # ----------------------- Text Upload Section -----------------------

    st.header("📄 Upload Text File")
    uploaded_file = st.file_uploader("Upload a .pdf or .docx file", type=['pdf', 'docx'])

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            try:
                if uploaded_file.type == "application/pdf":
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text()
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = Document(uploaded_file)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
                st.session_state.uploaded_text = text
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if st.session_state.uploaded_text:
        st.text_area("Extracted Text:", value=st.session_state.uploaded_text, height=300)

# ----------------------- Search Page -----------------------

elif page == "Search":
    st.title("🔍 Search Page")

    # ----------------------- Search Input Section -----------------------

    st.header("🔍 Enter Your Search Query")
    st.session_state.query_input = st.text_input("Search:", value=st.session_state.query_input)

    if st.button("Search"):
        if st.session_state.query_input.strip():
            with st.spinner("Searching..."):
                try:
                    response = aiml_client.chat_completions_create(
                        model="text-davinci-003",
                        messages=[
                            {"role": "user", "content": st.session_state.query_input}
                        ]
                    )
                    st.session_state.search_results = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    st.success("Search completed!")
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter a query to search.")

    # ----------------------- Search Results Section -----------------------

    if st.session_state.search_results:
        st.header("🔍 Search Results")
        st.write(st.session_state.search_results)

        if st.button("Save Search Results as PDF"):
            pdf = generate_pdf([st.session_state.query_input, st.session_state.search_results])
            pdf_file = BytesIO()
            pdf.output(pdf_file)
            pdf_file.seek(0)

            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="search_results.pdf",
                mime="application/pdf"
            )
