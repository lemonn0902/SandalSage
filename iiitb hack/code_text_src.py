import streamlit as st
import streamlit.components.v1 as components
import os
from audio_recorder_streamlit import audio_recorder
import pandas as pd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import google.generativeai as genai
from pydub import AudioSegment
import sqlite3
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
DB_PATH = "transcriptions.db"


genai.configure(api_key="<add ur api key>")# add ur gemini api key


st.markdown(""" 
    <style>
        .title { 
            font-size: 45px; 
            color: #FFFFFF; 
            text-align: center; 
            font-family: 'Lora', serif;
            font-weight: bold;
            background-color:rgb(170, 84, 134);
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            font-size: 30px; 
            color: #FFFFFF; 
            text-align: center; 
            margin-top: 20px;
            font-family: 'Roboto', sans-serif;
            background-color: #FC8F54;
            padding: 15px;
            border-radius: 8px;
        }
        .content {
            font-size: 18px; 
            text-align: center; 
            color: #444;
            font-family: 'Helvetica', sans-serif;
        }
        .card {
            background-color:rgb(170, 84, 134);
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
            margin: 20px;
        }
        .stButton button {
            background-color: rgb(170, 84, 134);
            color: white;
            border-radius: 15px;
            padding: 12px 30px;
            border: none;
            font-size: 18px;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: rgb(252, 143, 84);
        }
        .stApp {
            background-color: rgb(251, 244, 219);
        }
        .container-section {
            background-color: rgba(170, 84, 134, 0.3);
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
        }
        .content-section {
            background-color:rgb(253, 231, 187);
            border-radius: 8px;
            padding: 20px;
            margin-top: 10px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }
        .audio-section {
            background-color: rgb(251, 244, 219);
            border-radius: 10px;
            padding: 20px;
            margin-top: 10px;
        }
        .transcription-section {
            background-color:rgb(253, 231, 187);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }
        .search-result-section {
            background-color:rgb(253, 231, 187);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }
        .stTextArea textarea {
            background-color: rgb(253, 231, 187);
            border-radius: 8px;
            font-size: 18px;
            padding: 12px;
        }
        .hover-effect:hover {
            background-color: rgba(170, 84, 134, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)
botsonic_script = """
<script>
  (function (w, d, s, o, f, js, fjs) {
    w["botsonic_widget"] = o;
    w[o] =
      w[o] ||
      function () {
        (w[o].q = w[o].q || []).push(arguments);
      };
    (js = d.createElement(s)), (fjs = d.getElementsByTagName(s)[0]);
    js.id = o;
    js.src = f;
    js.async = 1;
    fjs.parentNode.insertBefore(js, fjs);
  })(window, document, "script", "Botsonic", "https://widget.botsonic.com/CDN/botsonic.min.js");
  Botsonic("init", {
    serviceBaseUrl: "https://api-azure.botsonic.ai",
    token: "d953474c-b19c-4a85-a531-2f90c1e660f7",
  });
</script>
"""

components.html(botsonic_script, height=0, width=0)


st.markdown('<div class="title">Sandalwood Cultivation Knowledge System</div>', unsafe_allow_html=True)


whisper_processor = WhisperProcessor.from_pretrained("./whisper-small-finetuned")
whisper_model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-finetuned")

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
def init_db():
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file TEXT,
            transcription TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
def save_to_db(audio_file, transcription,summary):
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transcriptions (audio_file, transcription, summary)
        VALUES (?, ?, ?)
    """, (audio_file, transcription,summary))
    conn.commit()
    conn.close()
def fetch_from_db():
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transcriptions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def load_and_process_transcriptions(file_path):
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8").fillna("")

        documents = [
            Document(
                page_content=row['transcription'],
                metadata={'file_name': row['file_name']}
            )
            for _, row in df.iterrows()
        ]

        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        
        
        for i, doc in enumerate(split_docs):
            doc.metadata['start_time'] = i * 10 
            doc.metadata['end_time'] = (i + 1) * 10  

        return split_docs, df
    except Exception as e:
        st.error(f"Error loading transcription file: {e}")
        return [], pd.DataFrame()


def initialize_retrievers(documents):
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever
    except Exception as e:
        st.error(f"Error initializing retrievers: {e}")
        raise


def save_audio_as_wav(audio_data):
    try:
        wav_path = "recorded_audio.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_data)
        return wav_path
    except Exception as e:
        raise ValueError(f"Error saving audio: {e}")


def transcribe_audio(audio_path):
    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
        input_features = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        raise ValueError(f"Error in transcription: {e}")

def extract_audio_chunks(audio_metadata, audio_dir="./Datasets/audio_files/"):
    """Extract relevant audio chunks from files based on metadata."""
    extracted_audio_files = []
    try:
        for segment in audio_metadata:
            file_name = segment.get('file_name')
            start_time = segment.get('start_time', 0) * 1000  
            end_time = segment.get('end_time', 0) * 1000      

            if not file_name:
                logging.warning("File name is missing in metadata.")
                continue

            if not file_name.endswith(".mp3"):
                file_name += ".mp3"

            audio_path = os.path.join(audio_dir, file_name)
            logging.info(f"Checking file path: {audio_path}")

            if not os.path.isfile(audio_path):
                logging.warning(f"File not found: {audio_path}")
                continue

            try:
                audio = AudioSegment.from_file(audio_path, format="mp3")
                audio_duration = len(audio)  

               
                if start_time >= audio_duration:
                    logging.warning(f"Start time exceeds audio duration for {file_name}. Setting start time to 0.")
                    start_time = 0 
                if end_time > audio_duration:
                    logging.warning(f"End time exceeds audio duration for {file_name}. Setting end time to the audio duration.")
                    end_time = audio_duration 

              
                if end_time <= start_time:
                    logging.warning(f"Invalid chunk timing for {file_name}: start={start_time}, end={end_time}. Skipping.")
                    continue

                chunk = audio[start_time:end_time]  
                chunk_path = f"extracted_{file_name}chunk{start_time // 1000}-{end_time // 1000}.mp3"
                chunk.export(chunk_path, format="mp3")
                extracted_audio_files.append(chunk_path)

            except Exception as audio_error:
                logging.error(f"Error processing file {file_name}: {audio_error}")

    except Exception as e:
        logging.error(f"Error extracting audio chunks: {e}")
    return extracted_audio_files



def get_answer(retriever, question):
    """Retrieve answer and relevant audio metadata for a question."""
    try:
        source_documents = retriever.get_relevant_documents(question)
        context = "\n\nRelevant context:\n"
        relevant_audio = []

        for i, doc in enumerate(source_documents, 1):
            context += f"\nDocument {i}:\n{doc.page_content}\n"
            start_time = doc.metadata.get('start_time', 'N/A')
            end_time = doc.metadata.get('end_time', 'N/A')
            relevant_audio.append({
                "start_time": start_time,
                "end_time": end_time,
                "file_name": doc.metadata.get('file_name', 'N/A')
            })
        
        return context, relevant_audio
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        raise




def summarize_text_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Summarize the following text:\n\n{text}")
        return response.text.strip()
    except Exception as e:
        st.error(f"Error during summarization with Gemini API: {e}")
        return "Summarization failed."


def main():
    init_db()
    transcription = ""
    
    summary = ""
    documents, transcriptions_df = load_and_process_transcriptions("transcriptions.csv")

    if not documents:
        st.error("No documents loaded. Please check your transcriptions file.")
        return

    retriever = initialize_retrievers(documents)

    with st.sidebar:
        st.markdown('<div class="header">Summary</div>', unsafe_allow_html=True)
        summary_container = st.empty()

    with st.container():
        st.markdown('<div class="header">Record Your Sandalwood Question</div>', unsafe_allow_html=True)

        with st.expander("Click to Record Your Question", expanded=True):
            audio_data = audio_recorder()
            if audio_data:
                st.audio(audio_data, format="audio/wav")
                st.markdown('<div class="content">Audio recorded successfully!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="content">No audio recorded. Please try again.</div>', unsafe_allow_html=True)

    if audio_data:
        audio_path = save_audio_as_wav(audio_data)
        transcription = transcribe_audio(audio_path)

        with st.container():
            st.markdown('<div class="transcription-section">Transcription of Your Question:</div>', unsafe_allow_html=True)
            st.write(transcription)

        if st.button("Get Answer"):
            if transcription:
                with st.spinner("generating summary..."):
                    answer, relevant_audio = get_answer(retriever, transcription)

            
                    st.markdown('<div class="search-result-section">Answer (Relevant Context):</div>', unsafe_allow_html=True)
                    st.write(answer)

            
                    with st.sidebar:
                        st.markdown('<div class="header">Relevant Audios</div>', unsafe_allow_html=True)
                        if relevant_audio:
                            extracted_audio = extract_audio_chunks(relevant_audio)
                            for audio_file in extracted_audio:
                                st.audio(audio_file, format="audio/mp3")
                        else:
                            st.warning("No relevant audio segments found.")


                    
                summary = summarize_text_gemini(answer)
                summary_container.markdown(
                    f"""
                    <div class="content-section">
                        <b>Summarized Answer:</b>
                        <p>{summary}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.warning("Transcription is empty. Please record again.")
        if st.button("Save to Database"):
            if transcription:
                audio_file_path = save_audio_as_wav(audio_data) 
                save_to_db(audio_file_path, transcription, summary)
                st.success("Data saved to the database!")
            else:
                st.error("Incomplete data. Please ensure transcription,and summary are available.")

        
       



if __name__ == "__main__":
    main()