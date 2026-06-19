import streamlit as st
from hailo_platform import VDevice
from hailo_platform.genai import LLM

MODEL_PATH = r"C:\Users\Test\hailo_ai\Qwen2-1.5B-Instruct.hef"
SYSTEM_PROMPT = "You are a helpful, concise assistant. Give short, direct answers unless the user asks for detail."

st.set_page_config(page_title="Hailo LLM Chat", page_icon="🤖", layout="centered")
st.title("🤖 Hailo LLM Chat")
st.caption("Powered by ASUS UGen300 · Qwen2-1.5B-Instruct · HailoRT 5.3.2")

@st.cache_resource(show_spinner="Loading model onto UGen300…")
def load_model():
    vdevice = VDevice()
    return LLM(vdevice, MODEL_PATH, "")

try:
    llm = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info(f"Make sure:\n- UGen300 is plugged in\n- HEF file exists at: {MODEL_PATH}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_turn = True

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    max_tokens = st.slider("Max response tokens", 64, 1024, 256, step=64)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.first_turn = True
        llm.clear_context()
        st.rerun()
    st.divider()
    st.markdown("**Device:** ASUS UGen300")
    st.markdown("**Model:** Qwen2-1.5B-Instruct")
    st.markdown("**Runtime:** HailoRT 5.3.2")
    st.markdown("**Chip:** Hailo-10H · 40 TOPS")

# Handle new input
if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def token_stream():
            # On first turn, include system prompt; model maintains context thereafter
            if st.session_state.first_turn:
                structured = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                st.session_state.first_turn = False
            else:
                structured = [{"role": "user", "content": prompt}]

            STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}
            with llm.generate(structured, max_generated_tokens=max_tokens) as gen:
                for token in gen:
                    if token in STOP_TOKENS:
                        break
                    yield token

        full_response = st.write_stream(token_stream())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
