import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
import soundfile as sf  # pip install soundfile

# Load environment variables from .env file
load_dotenv()
TEAM_API_KEY = os.getenv("TEAM_API_KEY")
if not TEAM_API_KEY:
    raise Exception("TEAM_API_KEY is not set in the environment.")

# Import your BioLLM class (ensure model.py is in the same directory)
from model import BioLLM

# Initialize BioLLM instance
bio_llm = BioLLM(api_key=TEAM_API_KEY)

# Streamlit app title and description
st.title("BioLLM Text & Audio Processing")
st.write("Select a mode (Text or Audio) from the sidebar, provide the required inputs, and click Process.")

# Sidebar for mode selection
mode = st.sidebar.radio("Select Input Mode", options=["Text", "Audio"])

if mode == "Text":
    st.header("Text Mode")
    text_input = st.text_area("Enter your text", height=150)
    source_language = st.text_input("Source Language", value="en")
    rag_query = st.text_input("RAG Query (optional)", value="")
    rag_category = st.text_input("RAG Category (optional)", value="general")
    
    if st.button("Process Text"):
        if not text_input.strip():
            st.error("Please provide some text.")
        else:
            with st.spinner("Processing text..."):
                # Process the text input using the BioLLM pipeline
                result = bio_llm.process_pipeline(
                    input_type="text",
                    text=text_input,
                    source_language=source_language,
                    target_language="en",
                    rag_query=rag_query if rag_query else text_input,
                    rag_category=rag_category if rag_category else "general"
                )
                # Get BioLLM result from the pipeline
                bio_llm_result = result.get("steps", {}).get("bio_llm_processing", {}).get("result", "")
                # If the source language is not English, translate the result back
                if source_language.lower() != "en":
                    translated = bio_llm.translate_text(
                        {"text": bio_llm_result, "source_language": "en"},
                        target_language=source_language
                    )
                    response_text = translated.get("translated_text", bio_llm_result)
                else:
                    response_text = bio_llm_result
                
                st.success("Processing complete!")
                st.subheader("Response")
                st.text_area("", response_text, height=200)

elif mode == "Audio":
    st.header("Audio Mode")
    st.write("Upload an audio file (preferably in WAV format) to process.")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
    source_language = st.text_input("Source Language", value="en")
    rag_query = st.text_input("RAG Query (optional)", value="")
    rag_category = st.text_input("RAG Category (optional)", value="general")
    
    if st.button("Process Audio"):
        if audio_file is None:
            st.error("Please upload an audio file.")
        else:
            with st.spinner("Processing audio..."):
                # Save the uploaded audio to a temporary .wav file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_filename = tmp.name
                    tmp.write(audio_file.read())
                
                # Process the audio using the BioLLM pipeline in speech mode
                result = bio_llm.process_pipeline(
                    input_type="audio",
                    audio_path=tmp_filename,
                    source_language=source_language,
                    target_language="en",
                    rag_query=rag_query,
                    rag_category=rag_category
                )
                
                
                # Extract the transcript and the BioLLM result from the processing steps
                transcript = result.get("steps", {}).get("input_processing", {}).get("text", "")
                bio_llm_result = result.get("steps", {}).get("bio_llm_processing", {}).get("result", "")
                
                # Remove the temporary file
                os.remove(tmp_filename)
                
                # If the source language is not English, translate the BioLLM result back
                if source_language.lower() != "en":
                    translated = bio_llm.translate_text(
                        {"text": bio_llm_result, "source_language": "en"},
                        target_language=source_language
                    )
                    response_text = translated.get("translated_text", bio_llm_result)
                else:
                    response_text = bio_llm_result
                
                st.success("Audio processing complete!")
                st.subheader("Transcript")
                st.text_area("", transcript, height=150)
                st.subheader("Response")
                st.text_area("", response_text, height=200)
