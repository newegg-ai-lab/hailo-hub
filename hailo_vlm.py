import streamlit as st
from hailo_platform import VDevice
from hailo_platform.genai import VLM
import cv2
import numpy as np
import hashlib
import atexit

MODEL_PATH = r"C:\Users\Test\hailo_ai\Qwen2-VL-2B-Instruct.hef"
SYSTEM_PROMPT = "You are a helpful vision assistant. Analyze images and answer questions concisely."
STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}

st.set_page_config(page_title="Hailo VLM", page_icon="👁️", layout="wide")
st.title("👁️ Hailo Vision Language Model")
st.caption("Powered by ASUS UGen300 · Qwen2-VL-2B-Instruct · HailoRT 5.3.2")


@st.cache_resource(show_spinner="Loading VLM onto UGen300… (this takes ~30s)")
def load_model():
    vdevice = VDevice()
    vlm = VLM(vdevice, MODEL_PATH, True)
    atexit.register(lambda: (vlm.release(), vdevice.release()))
    return vdevice, vlm


try:
    vdevice, vlm = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Make sure:\n- UGen300 is plugged in\n- No other Hailo app is running\n- Run the HailoRT MSI installer if needed")
    st.stop()

frame_shape = vlm.input_frame_shape()

if "vlm_messages" not in st.session_state:
    st.session_state.vlm_messages = []
    st.session_state.first_turn = True
    st.session_state.image_hash = None
    st.session_state.current_frame = None

with st.sidebar:
    st.header("Controls")
    max_tokens = st.slider("Max tokens", 64, 512, 256, step=64)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.vlm_messages = []
        st.session_state.first_turn = True
        st.session_state.image_hash = None
        vlm.clear_context()
        st.rerun()
    st.divider()
    ctx_used = vlm.get_context_usage_size()
    ctx_max = vlm.max_context_capacity()
    st.progress(ctx_used / ctx_max if ctx_max > 0 else 0, text=f"Context: {ctx_used}/{ctx_max}")
    st.divider()
    st.markdown("**Device:** ASUS UGen300")
    st.markdown("**Model:** Qwen2-VL-2B-Instruct")
    st.markdown("**Runtime:** HailoRT 5.3.2")
    st.markdown(f"**Frame:** {frame_shape[1]}×{frame_shape[0]}")

col_img, col_chat = st.columns([1, 1])

with col_img:
    st.subheader("Image")
    source = st.radio("Source", ["Upload file", "Camera"], horizontal=True)
    raw_frame = None

    if source == "Upload file":
        uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            img_bytes = uploaded.read()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
                h, w, _ = frame_shape
                raw_frame = cv2.resize(img_rgb, (w, h)).astype(np.uint8)
                if st.session_state.image_hash != img_hash:
                    st.session_state.image_hash = img_hash
                    st.session_state.current_frame = raw_frame
                    st.session_state.vlm_messages = []
                    st.session_state.first_turn = True
                    vlm.clear_context()
    else:
        cam_img = st.camera_input("Take photo")
        if cam_img:
            img_bytes = cam_img.read()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = frame_shape
                raw_frame = cv2.resize(img_rgb, (w, h)).astype(np.uint8)
                if st.session_state.image_hash != img_hash:
                    st.session_state.image_hash = img_hash
                    st.session_state.current_frame = raw_frame
                    st.session_state.vlm_messages = []
                    st.session_state.first_turn = True
                    vlm.clear_context()

    if st.session_state.current_frame is None:
        st.info("Upload or capture an image to get started.")

with col_chat:
    mode = st.radio("Mode", ["💬 Q&A", "🔍 Detect Objects"], horizontal=True)
    st.divider()

    if mode == "💬 Q&A":
        for msg in st.session_state.vlm_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the image…"):
            if st.session_state.current_frame is None:
                st.warning("Please provide an image first (upload or camera).")
            else:
                current_frame = st.session_state.current_frame
                st.session_state.vlm_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    def token_stream():
                        if st.session_state.first_turn:
                            structured = [
                                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                                {"role": "user", "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": prompt},
                                ]},
                            ]
                            st.session_state.first_turn = False
                            frames = [current_frame]
                        else:
                            structured = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                            frames = []

                        with vlm.generate(structured, frames=frames, max_generated_tokens=max_tokens) as gen:
                            for token in gen:
                                if token in STOP_TOKENS:
                                    break
                                yield token

                    full_response = st.write_stream(token_stream())

                st.session_state.vlm_messages.append({"role": "assistant", "content": full_response})

    else:  # Detect Objects mode
        DETECT_PROMPT = (
            "List every distinct object you can see in this image. "
            "For each object provide: name, approximate size (small/medium/large), "
            "and position (e.g. top-left, center, bottom-right). "
            "Format as a numbered list. Be concise."
        )
        if st.session_state.current_frame is None:
            st.info("Provide an image on the left, then click Detect.")
        else:
            if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
                vlm.clear_context()
                current_frame = st.session_state.current_frame
                with st.spinner("Running detection on UGen300…"):
                    structured = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a precise object detector. List objects concisely."}]},
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": DETECT_PROMPT},
                        ]},
                    ]
                    tokens = []
                    with vlm.generate(structured, frames=[current_frame], max_generated_tokens=max_tokens) as gen:
                        for token in gen:
                            if token in STOP_TOKENS:
                                break
                            tokens.append(token)
                    result = "".join(tokens)
                st.markdown(result)
                vlm.clear_context()
                st.session_state.first_turn = True
