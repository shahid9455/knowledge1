import streamlit as st
import requests
import base64
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# IBM Watson NLU Configuration
api_key_watson = 'YOUR_WATSON_API_KEY'
nlu_url = 'YOUR_WATSON_URL'

# Initialize IBM Watson NLU
authenticator = IAMAuthenticator(api_key_watson)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(nlu_url)

# AIML API Configuration
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

aiml_client = AIMLClient(api_key="YOUR_AIML_API_KEY", base_url="YOUR_AIML_BASE_URL")

# Define the filename where text will be stored
filename = 'text_storage_with_keywords.txt'

def save_text(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(keywords=KeywordsOptions(limit=5))
        ).get_result()
        
        keywords = [kw['text'] for kw in response['keywords']]
        keyword_string = ', '.join(keywords)
        
        with open(filename, 'a') as file:
            file.write(f"Text: {text}\nKeywords: {keyword_string}\n\n")
        
        st.success("Your input and extracted keywords have been saved successfully.")
    except Exception as e:
        st.error(f"An error occurred while processing the text: {str(e)}")

def search(query):
    refined_text = ""
    suggestions = []

    try:
        # Read the text file and get all keywords
        with open(filename, 'r') as file:
            lines = file.readlines()
            all_keywords = []
            all_texts = []

            for line in lines:
                if 'Keywords:' in line:
                    stored_keywords = [kw.strip() for kw in line.replace('Keywords:', '').split(',')]
                    all_keywords.extend(stored_keywords)
                if 'Text:' in line:
                    all_texts.append(line.replace('Text:', '').strip())

            # Provide suggestions based on close matches
            suggestions = difflib.get_close_matches(query, all_keywords, n=5, cutoff=0.6)

        # Search for matching texts
        matching_texts = [text for text in all_texts if any(kw.lower() in text.lower() for kw in suggestions)]

        if matching_texts:
            # Generate refined text using the AIML model
            response = aiml_client.chat_completions_create(
                model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                messages=[
                    {"role": "system", "content": "You are an AI assistant who knows everything."},
                    {"role": "user", "content": f"Refine the following text into a professional and polished summary without asking for additional data:\n\n{' '.join(matching_texts)}"}
                ]
            )
            refined_text = response['choices'][0]['message']['content'].strip()
            if not refined_text:
                refined_text = "No text generated."
        else:
            refined_text = f"No matching content found for '{query}'."
    except Exception as e:
        refined_text = f"An error occurred: {str(e)}"

    return refined_text, suggestions

def download_pdf(refined_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 100, "Here is a polished and professional summary:")
    text_object = c.beginText(100, height - 120)
    text_object.setFont("Helvetica", 12)
    text_object.setTextOrigin(100, height - 140)
    text_object.textLines(refined_text)
    c.drawText(text_object)
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer

# Voice recording UI with Streamlit
st.title("Knowledge and Experience Storage System")

st.header("Voice Recording")
st.markdown("""
    <button id="startRecording" class="btn btn-primary">Start Recording</button>
    <button id="stopRecording" class="btn btn-danger" disabled>Stop Recording</button>
    <p id="recordingStatus">Status: Idle</p>
    <script>
        let mediaRecorder;
        let audioChunks = [];
        const startButton = document.getElementById('startRecording');
        const stopButton = document.getElementById('stopRecording');
        const recordingStatus = document.getElementById('recordingStatus');

        startButton.addEventListener('click', async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            recordingStatus.textContent = 'Status: Recording...';
            startButton.disabled = true;
            stopButton.disabled = false;

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
        });

        stopButton.addEventListener('click', () => {
            mediaRecorder.stop();
            recordingStatus.textContent = 'Status: Idle';
            startButton.disabled = false;
            stopButton.disabled = true;

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                const base64Audio = await convertBlobToBase64(audioBlob);
                document.getElementById('audioData').value = base64Audio;
                document.getElementById('audioForm').submit();
            };
        });

        function convertBlobToBase64(blob) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result.split(',')[1]);
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        }
    </script>
""", unsafe_allow_html=True)

# Hidden form for audio data submission
st.markdown("""
    <form id="audioForm" method="POST" action="your_backend_endpoint">
        <input type="hidden" id="audioData" name="audioData">
    </form>
""", unsafe_allow_html=True)

# Save Text Section
st.header("Enter text to save:")
text = st.text_area("Text to save")
if st.button("Save Text"):
    if text:
        save_text(text)
    else:
        st.warning("No text entered to save.")

# Search Section
st.header("Enter keyword or topic to search:")
query = st.text_input("Keyword/Topic")
if st.button("Search"):
    if query:
        refined_text, suggestions = search(query)
        st.subheader("Refined Text:")
        st.write(refined_text)

        if st.button("Download PDF"):
            pdf = download_pdf(refined_text)
            st.download_button("Download Refined Text PDF", data=pdf, file_name="refined_text_summary.pdf")

        st.subheader("Suggestions:")
        st.write(suggestions)
    else:
        st.warning("Please enter your keyword to search.")

# Handle audio data saving on the backend
def save_audio(audio_data):
    # Process the base64 encoded audio data (save to file, send to API, etc.)
    st.success("Audio recorded and processed successfully.")
