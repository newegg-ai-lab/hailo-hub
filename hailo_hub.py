import streamlit as st
import atexit
import cv2
import numpy as np
import hashlib
import io
from hailo_platform import VDevice
from hailo_platform.genai import LLM, VLM, Speech2Text, Speech2TextTask

try:
    import scipy.io.wavfile
    from scipy.signal import resample as scipy_resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Paths ──────────────────────────────────────────────────────────────────
LLM_PATH = r"C:\Users\Test\hailo_ai\Qwen2-1.5B-Instruct.hef"
VLM_PATH = r"C:\Users\Test\hailo_ai\Qwen2-VL-2B-Instruct.hef"
S2T_PATH = r"C:\Users\Test\hailo_ai\Whisper-Small.hef"

LLM_SYSTEM  = "You are a helpful, concise assistant. Give short, direct answers unless the user asks for detail."
VLM_SYSTEM  = "You are a helpful vision assistant. Analyze images and answer questions concisely."
STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}
TARGET_RATE = 16000
DETECT_PROMPT = (
    "List every distinct object visible in this image. "
    "For each: name, size (small/medium/large), position in frame. "
    "Numbered list. Be concise."
)

# Plain ASCII keys — no emoji in keys to avoid widget-state encoding issues
MODE_OPTIONS = ["LLM Chat", "VLM Q&A", "VLM Detect", "Speech2Text"]
MODE_LABELS  = {
    "LLM Chat":    "LLM Chat   (Qwen2-1.5B)",
    "VLM Q&A":    "VLM Q&A   (Qwen2-VL-2B)",
    "VLM Detect": "VLM Detect (Qwen2-VL-2B)",
    "Speech2Text":"Speech2Text (Whisper-Small)",
}
MODE_TYPE = {
    "LLM Chat":    "llm",
    "VLM Q&A":    "vlm",
    "VLM Detect": "vlm",
    "Speech2Text":"speech",
}

# ── Page ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hailo Hub", page_icon="H", layout="wide")

# ── VDevice — one per session ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vdevice():
    vd = VDevice()
    atexit.register(vd.release)
    return vd

try:
    vdevice = get_vdevice()
except Exception as e:
    st.error(f"Cannot open UGen300: {e}")
    st.info(
        "Fix: re-run  C:\\Users\\Test\\Downloads\\HailoRT_5.3.2_windows_installer.msi  "
        "then refresh this page."
    )
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────
for _k, _v in {
    "active_mode":     "LLM Chat",
    "active_type":     None,
    "model":           None,
    "llm_messages":    [],
    "llm_first_turn":  True,
    "vlm_messages":    [],
    "vlm_first_turn":  True,
    "vlm_hash":        None,
    "vlm_frame":       None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Hailo Hub")
    st.caption("ASUS UGen300 | Hailo-10H | HailoRT 5.3.2")
    st.divider()

    st.markdown("**Select Demo**")
    selected = st.radio(
        label="demo",
        options=MODE_OPTIONS,
        format_func=lambda x: MODE_LABELS[x],
        label_visibility="collapsed",
        key="demo_radio",
    )

    st.divider()
    max_tokens = st.slider("Max tokens", 64, 512, 256, step=64)

    if selected in ("LLM Chat", "VLM Q&A"):
        if st.button("Clear chat", use_container_width=True):
            if selected == "LLM Chat":
                st.session_state.llm_messages  = []
                st.session_state.llm_first_turn = True
                if st.session_state.active_type == "llm" and st.session_state.model:
                    st.session_state.model.clear_context()
            else:
                st.session_state.vlm_messages  = []
                st.session_state.vlm_first_turn = True
                st.session_state.vlm_hash       = None
                if st.session_state.active_type == "vlm" and st.session_state.model:
                    st.session_state.model.clear_context()
            st.rerun()

    st.divider()
    target_type = MODE_TYPE[selected]
    if st.session_state.active_type == target_type:
        st.success(f"Loaded: {selected}")
    else:
        st.info("Click a demo to load it")
    st.caption("Chip: Hailo-10H 40 TOPS | RAM: 8 GB")

# ── Model switching ────────────────────────────────────────────────────────
target_type = MODE_TYPE[selected]

if target_type != st.session_state.active_type:
    placeholder = st.empty()

    # Release old
    if st.session_state.model is not None:
        with placeholder.container():
            with st.spinner("Releasing previous model..."):
                st.session_state.model.release()
                st.session_state.model = None

    # Load new
    labels = {"llm": "Qwen2-1.5B-Instruct", "vlm": "Qwen2-VL-2B-Instruct", "speech": "Whisper-Small"}
    with placeholder.container():
        with st.spinner(f"Loading {labels[target_type]} onto UGen300...  (30-60 s first time)"):
            try:
                if target_type == "llm":
                    st.session_state.model          = LLM(vdevice, LLM_PATH, "")
                    st.session_state.llm_messages   = []
                    st.session_state.llm_first_turn = True
                elif target_type == "vlm":
                    st.session_state.model          = VLM(vdevice, VLM_PATH, True)
                    st.session_state.vlm_messages   = []
                    st.session_state.vlm_first_turn = True
                    st.session_state.vlm_hash       = None
                    st.session_state.vlm_frame      = None
                elif target_type == "speech":
                    st.session_state.model = Speech2Text(vdevice, S2T_PATH)
            except Exception as e:
                placeholder.error(f"Failed to load model: {e}\n\nRe-run the HailoRT MSI installer then refresh.")
                st.stop()

    st.session_state.active_type = target_type
    st.session_state.active_mode = selected
    placeholder.empty()
    st.rerun()

st.session_state.active_mode = selected
model = st.session_state.model

# ══════════════════════════════════════════════════════════════════════════
# LLM CHAT
# ══════════════════════════════════════════════════════════════════════════
if selected == "LLM Chat":
    st.header("LLM Chat")
    st.caption("Qwen2-1.5B-Instruct | text generation on Hailo-10H")

    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.llm_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            def _llm_stream():
                if st.session_state.llm_first_turn:
                    msgs = [
                        {"role": "system", "content": LLM_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ]
                    st.session_state.llm_first_turn = False
                else:
                    msgs = [{"role": "user", "content": prompt}]
                with model.generate(msgs, max_generated_tokens=max_tokens) as gen:
                    for token in gen:
                        if token in STOP_TOKENS:
                            break
                        yield token
            reply = st.write_stream(_llm_stream())
        st.session_state.llm_messages.append({"role": "assistant", "content": reply})

# ══════════════════════════════════════════════════════════════════════════
# VLM Q&A  +  VLM DETECT  (share the same VLM model)
# ══════════════════════════════════════════════════════════════════════════
elif selected in ("VLM Q&A", "VLM Detect"):
    frame_shape = model.input_frame_shape()   # (H, W, C)
    H, W, _ = frame_shape

    col_img, col_right = st.columns([1, 1])

    with col_img:
        if selected == "VLM Q&A":
            st.header("VLM Q&A")
            st.caption("Qwen2-VL-2B | vision + language on Hailo-10H")
        else:
            st.header("VLM Detect")
            st.caption("Qwen2-VL-2B | object detection via VLM on Hailo-10H")

        src = st.radio("Image source", ["Upload file", "Camera"], horizontal=True, key="img_src")

        if src == "Upload file":
            up = st.file_uploader("Choose image", type=["jpg","jpeg","png","bmp","webp"])
            if up:
                raw   = up.read()
                ihash = hashlib.md5(raw).hexdigest()
                arr   = np.frombuffer(raw, np.uint8)
                bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                    frame = cv2.resize(rgb, (W, H)).astype(np.uint8)
                    if st.session_state.vlm_hash != ihash:
                        st.session_state.vlm_hash      = ihash
                        st.session_state.vlm_frame     = frame
                        st.session_state.vlm_messages  = []
                        st.session_state.vlm_first_turn= True
                        model.clear_context()
        else:
            cam = st.camera_input("Take photo")
            if cam:
                raw   = cam.read()
                ihash = hashlib.md5(raw).hexdigest()
                arr   = np.frombuffer(raw, np.uint8)
                bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(rgb, (W, H)).astype(np.uint8)
                    if st.session_state.vlm_hash != ihash:
                        st.session_state.vlm_hash      = ihash
                        st.session_state.vlm_frame     = frame
                        st.session_state.vlm_messages  = []
                        st.session_state.vlm_first_turn= True
                        model.clear_context()

        ctx_used = model.get_context_usage_size()
        ctx_max  = model.max_context_capacity()
        if ctx_max > 0:
            st.progress(min(ctx_used / ctx_max, 1.0),
                        text=f"Context: {ctx_used}/{ctx_max} tokens")

        if st.session_state.vlm_frame is None:
            st.info("Upload or take a photo to get started.")

    with col_right:
        frame_now = st.session_state.vlm_frame

        # ── Q&A ───────────────────────────────────────────────────────────
        if selected == "VLM Q&A":
            for msg in st.session_state.vlm_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask about the image..."):
                if frame_now is None:
                    st.warning("Provide an image first.")
                else:
                    st.session_state.vlm_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        def _vlm_stream():
                            if st.session_state.vlm_first_turn:
                                msgs = [
                                    {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM}]},
                                    {"role": "user",   "content": [
                                        {"type": "image"},
                                        {"type": "text", "text": prompt},
                                    ]},
                                ]
                                frames = [frame_now]
                                st.session_state.vlm_first_turn = False
                            else:
                                msgs   = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                                frames = []
                            with model.generate(msgs, frames=frames, max_generated_tokens=max_tokens) as gen:
                                for token in gen:
                                    if token in STOP_TOKENS:
                                        break
                                    yield token
                        reply = st.write_stream(_vlm_stream())
                    st.session_state.vlm_messages.append({"role": "assistant", "content": reply})

        # ── Detect ────────────────────────────────────────────────────────
        else:
            st.markdown("### Object Detection")
            st.markdown("Uses the VLM on-device to identify and locate every object in the image.")
            if frame_now is None:
                st.info("Provide an image on the left first.")
            else:
                if st.button("Detect Objects", type="primary", use_container_width=True):
                    model.clear_context()
                    with st.spinner("Running on UGen300..."):
                        msgs = [
                            {"role": "system", "content": [{"type": "text", "text": "You are a precise object detector."}]},
                            {"role": "user",   "content": [
                                {"type": "image"},
                                {"type": "text", "text": DETECT_PROMPT},
                            ]},
                        ]
                        toks = []
                        with model.generate(msgs, frames=[frame_now], max_generated_tokens=max_tokens) as gen:
                            for token in gen:
                                if token in STOP_TOKENS:
                                    break
                                toks.append(token)
                    st.markdown("".join(toks))
                    model.clear_context()
                    st.session_state.vlm_first_turn = True

# ══════════════════════════════════════════════════════════════════════════
# SPEECH2TEXT
# ══════════════════════════════════════════════════════════════════════════
elif selected == "Speech2Text":
    st.header("Speech to Text")
    st.caption("Whisper-Small | transcription & translation on Hailo-10H")

    if not HAS_SCIPY:
        st.error("scipy missing — run:  pip install scipy  in hailo_venv")
        st.stop()

    col_ctl, col_out = st.columns([1, 1])

    with col_ctl:
        task_choice = st.radio("Task", ["Transcribe", "Translate to English"])
        language    = st.selectbox("Source language",
            ["auto-detect","en","zh","es","fr","de","ja","ko","ar","pt","ru","it"])
        show_segs   = st.checkbox("Show timestamped segments", value=True)
        st.divider()
        audio_val = st.audio_input("Record audio")
        if audio_val:
            st.audio(audio_val)

    with col_out:
        if audio_val:
            if st.button("Transcribe", type="primary", use_container_width=True):
                wav_bytes = audio_val.read()
                with st.spinner("Converting audio..."):
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
                        n = int(len(audio) * TARGET_RATE / rate)
                        audio = scipy_resample(audio, n).astype(np.float32)

                s2t_task = Speech2TextTask.TRANSCRIBE if task_choice == "Transcribe" else Speech2TextTask.TRANSLATE
                lang_arg = None if language == "auto-detect" else language

                with st.spinner("Running Whisper on UGen300..."):
                    try:
                        segments = (
                            model.generate_all_segments(audio, task=s2t_task, language=lang_arg)
                            if lang_arg else
                            model.generate_all_segments(audio, task=s2t_task)
                        )
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        st.stop()

                full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
                st.success("Done!")
                st.text_area("Full transcription", full_text, height=140)
                if show_segs and segments:
                    st.subheader("Segments")
                    for i, seg in enumerate(segments, 1):
                        st.markdown(
                            f"**{i}.** `[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]`  {seg.text.strip()}"
                        )
        else:
            st.info("Record audio on the left, then click Transcribe.")
