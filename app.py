import streamlit as st
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydub import AudioSegment
import torch
import tempfile
import os
from io import BytesIO
# PyAnnote imports
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

st.title("🌐 Multilingual Audio Summarizer with Speaker Diarization")

# Check if HuggingFace token is available
if not hf_token:
    st.error("⚠️ HuggingFace token not found! Please set HUGGINGFACE_HUB_TOKEN in your .env file.")
    st.info("You need to accept the user agreement for pyannote/speaker-diarization model on HuggingFace Hub.")
    st.stop()

# Upload audio
uploaded_file = st.file_uploader(
    "Upload audio", 
    type=["mp3", "wav", "m4a", "ogg", "opus"]
)

target_lang = st.selectbox("Target language for summary", ["English", "Chinese"])

if uploaded_file:
    with st.spinner("Processing audio file..."):
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save and convert audio to WAV
            if uploaded_file.type == "audio/wav":
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            else:
                # Convert to WAV format
                audio_data = uploaded_file.getvalue()
                with tempfile.NamedTemporaryFile(delete=False) as input_temp:
                    input_temp.write(audio_data)
                    input_temp.flush()
                    
                    audio = AudioSegment.from_file(BytesIO(audio_data))
                    audio.export(temp_path, format="wav")

            # 1️⃣ Speaker Diarization
            st.info("🔍 Performing speaker diarization...")
            try:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                
                if diarization_pipeline is None:
                    raise ValueError("Failed to load diarization pipeline")
                    
                diarization = diarization_pipeline(temp_path)
                st.success("✅ Speaker diarization completed!")
                
            except Exception as e:
                st.error(f"❌ Speaker diarization failed: {str(e)}")
                st.warning("Continuing without speaker diarization...")
                diarization = None

            # 2️⃣ Transcription with Faster Whisper
            st.info("🎙️ Transcribing audio...")
            try:
                whisper_model = WhisperModel("large-v3", device="cpu")  # Use CPU for better compatibility
                segments, info = whisper_model.transcribe(temp_path, beam_size=5)
                
                # Convert segments to list for multiple iterations
                segments_list = list(segments)
                
                st.success(f"✅ Transcription completed! Detected language: {info.language}")
                
            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
                st.stop()

            # Combine transcription with speaker labels
            transcript = ""
            if diarization:
                for seg in segments_list:
                    # Find speaker label from diarization
                    speaker_label = "Unknown"
                    seg_start = seg.start
                    seg_end = seg.end
                    
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        # Check if segment overlaps with speaker turn
                        if (seg_start >= turn.start and seg_start <= turn.end) or \
                           (seg_end >= turn.start and seg_end <= turn.end) or \
                           (seg_start <= turn.start and seg_end >= turn.end):
                            speaker_label = speaker
                            break
                    
                    transcript += f"[{speaker_label}] {seg.text.strip()}\n"
            else:
                # Without diarization, just use the transcription
                for seg in segments_list:
                    transcript += f"[Speaker] {seg.text.strip()}\n"
            
            st.subheader("📝 Transcript with Speakers")
            st.text_area("Transcript", transcript, height=300)

            # 3️⃣ Optional translation
            translated = transcript
            target_lang_code = target_lang.lower()
            
            if target_lang_code != info.language and target_lang_code != "english":
                st.info("🔄 Translating transcript...")
                try:
                    # Map language codes
                    lang_map = {
                        "chinese": "zh",
                        "english": "en"
                    }
                    
                    target_code = lang_map.get(target_lang_code, target_lang_code[:2])
                    source_code = info.language
                    
                    translator_model = f"Helsinki-NLP/opus-mt-{source_code}-{target_code}"
                    translator_pipeline = pipeline("translation", model=translator_model)
                    
                    translated_text = ""
                    for line in transcript.split("\n"):
                        if line.strip():
                            if "]" in line:
                                speaker_tag = line.split("]")[0] + "]"
                                text = "]".join(line.split("]")[1:]).strip()
                            else:
                                speaker_tag = "[Speaker]"
                                text = line.strip()
                            
                            if text:
                                try:
                                    translated_line = translator_pipeline(text)[0]['translation_text']
                                    translated_text += f"{speaker_tag} {translated_line}\n"
                                except:
                                    translated_text += f"{speaker_tag} {text}\n"
                    
                    translated = translated_text
                    st.success("✅ Translation completed!")
                    
                except Exception as e:
                    st.warning(f"⚠️ Translation failed: {str(e)}. Using original transcript.")

            # 4️⃣ Summarization
            st.subheader("📊 Summary")
            st.info("🤖 Generating summary...")
            
            try:
                # Use a smaller, more accessible model for summarization
                summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Clean the text for summarization (remove speaker tags)
                clean_text = ""
                for line in translated.split("\n"):
                    if line.strip():
                        if "]" in line:
                            text = "]".join(line.split("]")[1:]).strip()
                        else:
                            text = line.strip()
                        clean_text += text + " "
                
                # Limit text length for summarization
                max_length = 1024  # BART's max input length
                if len(clean_text) > max_length:
                    clean_text = clean_text[:max_length]
                
                if clean_text.strip():
                    summary = summarizer(
                        clean_text,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    
                    st.success("✅ Summary generated!")
                    st.write("**Summary:**")
                    st.write(summary[0]['summary_text'])
                else:
                    st.warning("⚠️ No text available for summarization.")
                
            except Exception as e:
                st.error(f"❌ Summarization failed: {str(e)}")
                st.info("You might want to try a different summarization approach or check your model installation.")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
