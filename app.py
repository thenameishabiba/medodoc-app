
import io
import os
import numpy as np
import soundfile as sf
import streamlit as st
import requests
import torch
import torchaudio
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from datetime import datetime

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # set in env or .streamlit/secrets.toml
SUPABASE_URL   = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "")

DEFAULT_CHUNK_SEC = 6
DEFAULT_OVERLAP_SEC = 1.0

ALL_HEADERS = [
    "Medical History Update", "Intraoral Photos", "Chief Complaint",
    "Extra- and Intra-Oral Exams", "Radiographs", "Diagnosis",
    "Discussion", "Treatment Plan", "Procedure", "Next Visit", "Other"
]

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="MedoDoc — AI Medical Scribe", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main {background: #f8fafc;}
    h1, h2, h3 {color: #1e40af;}
    .summary-field {background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.6rem; margin-bottom: 1.2rem;}
    .summary-label {color: #1e40af; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.8rem; border-bottom: 2px solid #bfdbfe;}
    .transcript-box {background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 12px; padding: 1.3rem; height: 420px; overflow-y: auto;}
    .chunk-done {background: white; border-left: 5px solid #22c55e; padding: 0.8rem 1rem; margin: 8px 0; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ========================= SESSION STATE =========================
for key in ["transcript", "translated", "duration_sec", "summary_dict", "audio_bytes"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ["transcript", "translated"] else \
                                0.0 if key == "duration_sec" else \
                                {} if key == "summary_dict" else None

# ========================= LOAD WHISPER =========================
@st.cache_resource(show_spinner=False)
def load_whisper():
    return WhisperModel("large-v3", device="CPU", compute_type="int8")

whisper_model = load_whisper()

# ========================= AUDIO HELPERS =========================
def load_audio(audio_bytes):
    buffer = io.BytesIO(audio_bytes)
    audio_array, sr = sf.read(buffer)
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    return np.array(audio_array, dtype=np.float32), int(sr)

def preprocess_audio(audio_array, sr):
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
        sr = 16000
    audio_array = audio_tensor.squeeze(0).numpy()
    if np.max(np.abs(audio_array)) > 1.0:
        audio_array /= np.max(np.abs(audio_array))
    return audio_array, sr

def split_into_chunks(audio_array, sr, chunk_sec, overlap_sec):
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while start < len(audio_array):
        end = min(start + chunk_samples, len(audio_array))
        chunk = audio_array[start:end]
        if len(chunk) >= int(0.8 * sr):
            chunks.append((start / sr, chunk))
        start += step
    return chunks

def transcribe_chunk(chunk_array, chunk_start_sec):
    try:
        segments, info = whisper_model.transcribe(
            chunk_array, language="hi", beam_size=5, temperature=0.0,
            vad_filter=True, vad_parameters={"threshold": 0.62},
            initial_prompt="Doctor and patient speaking in Hinglish.",
            no_repeat_ngram_size=3, condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return chunk_start_sec, text, info.duration if info else 0.0
    except Exception as e:
        return chunk_start_sec, f"[Error: {e}]", 0.0

def _strip_overlap(prev, new):
    p = prev.split()
    n = new.split()
    for i in range(min(8, len(p), len(n)), 0, -1):
        if p[-i:] == n[:i]:
            return " ".join(n[i:]).strip()
    return new.strip()

def translate_and_clean(text):
    if not text: return ""
    try:
        trans = GoogleTranslator(source="auto", target="en").translate(text[:4000])
        corrections = {"parasitamol": "Paracetamol", "nousiy": "nausea", "vommiting": "vomiting",
                       "hedache": "headache", "fiver": "fever", "ulti": "vomiting", "chakkar": "dizziness"}
        for k, v in corrections.items():
            trans = trans.replace(k, v)
        return trans
    except:
        return text

def summarize_gemini(text, headers):
    if not GEMINI_API_KEY:
        return {h: "GEMINI_API_KEY not configured" for h in headers}
    prompt = f"""You are a professional medical scribe. Create structured clinical notes from this conversation.
Conversation:
{text[:4800]}
Return ONLY these exact fields:
""" + "\n".join(f"- {h}:" for h in headers) + """
Use 'Not provided.' if information is missing."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2}
        }, timeout=60)
        if response.status_code == 200:
            raw = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            result = {h: "Not provided." for h in headers}
            current = None
            buffer = []
            for line in raw.split("\n"):
                for h in headers:
                    if line.strip().startswith(h + ":") or line.strip().startswith("- " + h + ":"):
                        if current:
                            result[current] = "\n".join(buffer).strip() or "Not provided."
                        current = h
                        buffer = [line.split(":", 1)[1].strip()] if ":" in line else []
                        break
                else:
                    if current and line.strip():
                        buffer.append(line.strip())
            if current:
                result[current] = "\n".join(buffer).strip() or "Not provided."
            return result
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return {h: "Failed to generate summary." for h in headers}

def fmt_duration(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"

def save_to_supabase(audio_bytes, transcript, translated, summary_dict, duration):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, "Supabase not configured"
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        filename = f"medodoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_url = ""
        if audio_bytes:
            sb.storage.from_("audio_clips").upload(filename, audio_bytes, {"content-type": "audio/wav"})
            audio_url = sb.storage.from_("audio_clips").get_public_url(filename)
        data = {
            "audio_url": audio_url,
            "transcribe_text": transcript,
            "translated_text": translated,
            "summary": str(summary_dict),
            "language": "en",
            "duration": duration,
        }
        r = sb.table("medical_conversations").insert(data).execute()
        record_id = r.data[0]["id"] if r.data else None
        return record_id, None
    except Exception as e:
        return None, str(e)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1.5rem 0;'>
        <div style='font-size:2.6rem'>🩺</div>
        <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;color:#000000;'>MedoDoc</div>
        <div style='color:#64748b;font-size:0.85rem;'>AI Medical Scribe</div>
    </div>""", unsafe_allow_html=True)
    chunk_sec = st.slider("Chunk size (seconds)", 4, 12, DEFAULT_CHUNK_SEC, 1)
    overlap_sec = st.slider("Overlap (seconds)", 0.5, 2.0, DEFAULT_OVERLAP_SEC, 0.1)
    st.markdown("**Examination Template**")
    selected_headers = []
    for h in ALL_HEADERS:
        default = h in [    "Medical History Update",
    "Intraoral Photos",
    "Chief Complaint",
    "Extra- and Intra-Oral Exams",
    "Radiographs",
    "Diagnosis",
    "Discussion",
    "Treatment Plan",
    "Procedure",
    "Next Visit",
    "Other"]
        if st.checkbox(h, value=default, key=f"hdr_{h}"):
            selected_headers.append(h)
  #  selected_headers = [h for h in ALL_HEADERS if st.checkbox(h, value=True)]
    auto_translate = st.toggle("Auto translate to English", value=True)

st.title("MedoDoc — AI Medical Scribe")
st.markdown("")

tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])

# ====================== UPLOAD TAB ======================
with tab_upload:
    uploaded = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        st.audio(uploaded)
        st.session_state.audio_bytes = uploaded.getvalue()
        st.success("Audio uploaded successfully!")

# ====================== RECORD TAB (SIMPLE & WORKING) ======================
with tab_record:
    st.subheader("🎙️ Record Doctor-Patient Conversation")
    st.info("Click the microphone button below → speak → click stop. Audio will be saved automatically.")

    recorded_audio = st.audio_input("Record Audio")

    if recorded_audio is not None:
        st.session_state.audio_bytes = recorded_audio.getvalue()
        st.success("✅ Recording saved successfully!")
        st.audio(recorded_audio, format="audio/wav")

# ====================== TRANSCRIPTION ======================
if st.button("▶ Start Transcription", type="primary", use_container_width=True):
    if not st.session_state.get("audio_bytes"):
        st.error("❌ Please record or upload audio first!")
        st.stop()

    st.subheader("📝 Live Transcription")
    placeholder = st.empty()
    progress_bar = st.progress(0)
    status = st.empty()
    status.info("🔄 Transcribing...")

    audio_array, sr = load_audio(st.session_state.audio_bytes)
    audio_array, sr = preprocess_audio(audio_array, sr)
    chunks = split_into_chunks(audio_array, sr, chunk_sec, overlap_sec)

    chunk_results = []
    total_duration = 0.0

    for i, (start_sec, chunk) in enumerate(chunks):
        _, text, dur = transcribe_chunk(chunk, start_sec)
        total_duration += dur
        if text.strip():
            if chunk_results:
                text = _strip_overlap(chunk_results[-1][1], text)
            chunk_results.append((start_sec, text))

        html = "".join(f'<div class="chunk-done"><span class="timestamp">[{fmt_duration(s)}]</span> {t}</div>'
                       for s, t in chunk_results)
        placeholder.markdown(f'<div class="transcript-box">{html}</div>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(chunks))

    st.session_state.transcript = " ".join(t for _, t in chunk_results).strip()
    st.session_state.duration_sec = total_duration

    if auto_translate and st.session_state.transcript:
        with st.spinner("Translating to English..."):
            st.session_state.translated = translate_and_clean(st.session_state.transcript)

    status.success(f"✅ Transcription completed! Duration: {fmt_duration(total_duration)}")

# Show transcript
if st.session_state.get("transcript"):
    st.subheader("📝 Full Transcript")
    st.text_area("Transcript", st.session_state.transcript, height=180)

# ====================== SUMMARY & SAVE ======================
if st.button("✨ Generate Clinical Summary & Save to Supabase", type="primary", use_container_width=True):
    if not st.session_state.get("transcript"):
        st.error("Please perform transcription first!")
    else:
        with st.spinner("Generating summary using Gemini..."):
            text = st.session_state.translated or st.session_state.transcript
            st.session_state.summary_dict = summarize_gemini(text, selected_headers)

        with st.spinner("Saving to Supabase..."):
            record_id, error = save_to_supabase(
                st.session_state.get("audio_bytes"),
                st.session_state.transcript,
                st.session_state.get("translated", ""),
                st.session_state.summary_dict,
                st.session_state.duration_sec
            )
            if error:
                st.error(f"❌ Save failed: {error}")
            else:
                st.success(f"✅ Successfully saved! Record ID: {record_id}")

# Display Summary
if st.session_state.get("summary_dict"):
    st.subheader("🏥 Clinical Summary")
    for header, content in st.session_state.summary_dict.items():
        st.markdown(f"""
        <div class="summary-field">
            <div class="summary-label">{header}</div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)

