import streamlit as st
from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask
import numpy as np
import io
import atexit

try:
    import scipy.io.wavfile
    from scipy.signal import resample as scipy_resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODEL_PATH = r"C:\Users\Test\hailo_ai\Whisper-Small.hef"
TARGET_RATE = 16000

st.set_page_config(page_title="Hailo Speech2Text", page_icon="🎤", layout="centered")
st.title("🎤 Hailo Speech to Text")
st.caption("Powered by ASUS UGen300 · Whisper-Small · HailoRT 5.3.2")

if not HAS_SCIPY:
    st.error("scipy is required. Run: pip install scipy")
    st.stop()


@st.cache_resource(show_spinner="Loading Whisper onto UGen300…")
def load_model():
    vdevice = VDevice()
    s2t = Speech2Text(vdevice, MODEL_PATH)
    atexit.register(lambda: (s2t.release(), vdevice.release()))
    return vdevice, s2t


try:
    vdevice, s2t = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Make sure:\n- UGen300 is plugged in\n- No other Hailo app is running\n- Run the HailoRT MSI installer if needed")
    st.stop()

with st.sidebar:
    st.header("Settings")
    task_choice = st.radio("Task", ["Transcribe", "Translate to English"], horizontal=False)
    language = st.selectbox(
        "Source language",
        ["auto-detect", "en", "zh", "es", "fr", "de", "ja", "ko", "ar", "pt", "ru", "it"],
    )
    show_segments = st.checkbox("Show timestamped segments", value=True)
    st.divider()
    st.markdown("**Device:** ASUS UGen300")
    st.markdown("**Model:** Whisper-Small")
    st.markdown("**Runtime:** HailoRT 5.3.2")
    st.markdown("**Input:** 16 kHz mono float32")


def wav_to_float32_16k(wav_bytes: bytes) -> np.ndarray:
    rate, data = scipy.io.wavfile.read(io.BytesIO(wav_bytes))

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data.copy()
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if rate != TARGET_RATE:
        n_samples = int(len(audio) * TARGET_RATE / rate)
        audio = scipy_resample(audio, n_samples).astype(np.float32)

    return audio


st.info("Click the microphone button to record, then click **Transcribe**.")
audio_value = st.audio_input("Record audio")

if audio_value:
    st.audio(audio_value)
    wav_bytes = audio_value.read()

    if st.button("Transcribe", type="primary", use_container_width=True):
        with st.spinner("Processing audio…"):
            try:
                audio = wav_to_float32_16k(wav_bytes)
            except Exception as e:
                st.error(f"Audio conversion failed: {e}")
                st.stop()

            s2t_task = (
                Speech2TextTask.TRANSCRIBE
                if task_choice == "Transcribe"
                else Speech2TextTask.TRANSLATE
            )
            lang_arg = None if language == "auto-detect" else language

            try:
                with st.spinner("Running Whisper inference…"):
                    if lang_arg:
                        segments = s2t.generate_all_segments(audio, task=s2t_task, language=lang_arg)
                    else:
                        segments = s2t.generate_all_segments(audio, task=s2t_task)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        st.success("Transcription complete!")

        st.subheader("Result")
        st.text_area("Full text", full_text, height=120)

        if show_segments and segments:
            st.subheader("Segments")
            for i, seg in enumerate(segments, 1):
                ts = f"[{seg.start_sec:.1f}s – {seg.end_sec:.1f}s]"
                st.markdown(f"**{i}.** `{ts}` {seg.text.strip()}")
