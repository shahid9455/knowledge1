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
stt_api_key = 'LZIYjqxXb_9SzB__zoKP3B_RsciBfjrlqpeezDC-HbRD'
stt_url = 'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/ef13fb51-cbd7-488c-83ce-1363fd782882'

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
page = st.sidebar.radio("Go to", ["Input", "Search", "Audio Recording"])

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

                    with open(filename, 'a', encoding='utf-8') as file:
                        file.write(f"Text: {text_input}\nKeywords: {keyword_string}\n\n")

                    st.session_state.transcribed_text = ""
                    st.success("Text and keywords saved successfully!")
                except Exception as e:
                    st.error(f"Error processing text: {e}")
        else:
            st.warning("Please enter some text before saving.")

    # ----------------------- File Upload Section -----------------------

    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            try:
                if uploaded_file.type == "application/pdf":
                    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                        text = ""
                        for page in pdf:
                            text += page.get_text()
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = Document(uploaded_file)
                    text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    st.error("Unsupported file type!")
                    text = ""

                if text:
                    st.session_state.uploaded_text = text
                    st.text_area("Extracted Text:", value=text, height=300)

                    if st.button("Save Extracted Text"):
                        with st.spinner("Extracting keywords and saving text..."):
                            try:
                                response = nlu.analyze(
                                    text=text,
                                    features=Features(
                                        keywords=KeywordsOptions(limit=15)
                                    )
                                ).get_result()

                                keywords = [kw['text'] for kw in response.get('keywords', [])]
                                keyword_string = ', '.join(keywords)

                                with open(filename, 'a', encoding='utf-8') as file:
                                    file.write(f"Text: {text}\nKeywords: {keyword_string}\n\n")

                                st.session_state.uploaded_text = ""
                                st.success("Extracted text and keywords saved successfully!")
                            except Exception as e:
                                st.error(f"Error processing extracted text: {e}")
                else:
                    st.error("No text extracted from the uploaded file.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

# ----------------------- Search Page -----------------------

elif page == "Search":
    st.title("🔍 Search Page")

    st.header("🔎 Search Stored Knowledge")
    query_input = st.text_input("Enter keywords to search:")

    if st.button("Search"):
        if query_input.strip():
            with st.spinner("Searching for relevant texts..."):
                try:
                    query_keywords = [kw.strip().lower() for kw in query_input.split(',')]

                    all_keywords = []
                    with open(filename, 'r', encoding='utf-8') as file:
                        stored_texts = file.read().split('\n\n')
                        for text_block in stored_texts:
                            if "Keywords:" in text_block:
                                keyword_line = text_block.split("Keywords:")[1].strip().lower()
                                all_keywords.append((text_block, keyword_line.split(', ')))

                    search_results = [text for text, kw_list in all_keywords if any(kw in kw_list for kw in query_keywords)]

                    st.session_state.search_results = search_results

                    if search_results:
                        st.success("Search completed! Relevant texts found:")
                        for result in search_results:
                            st.markdown(f"<pre>{result}</pre>", unsafe_allow_html=True)
                    else:
                        st.warning("No relevant texts found for the given keywords.")
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter some keywords to search.")

# ----------------------- Audio Recording Page -----------------------

elif page == "Audio Recording":
    st.title("🎤 Audio Recording")

    st.header("🎤 Record and Save Text")

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

    st.header("📝 Transcribed Text")
    text_input = st.text_area("Text from transcription:", value=st.session_state.transcribed_text, height=200)

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

                    with open(filename, 'a', encoding='utf-8') as file:
                        file.write(f"Text: {text_input}\nKeywords: {keyword_string}\n\n")

                    st.session_state.transcribed_text = ""
                    st.success("Text and keywords saved successfully!")
                except Exception as e:
                    st.error(f"Error processing text: {e}")
        else:
            st.warning("Please enter some text before saving.")
