import streamlit as st
from main import ask_agent
import time
import json
from datetime import datetime
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Constants ----
MAX_CHAT_HISTORY = 100
TYPING_SPEED = 0.03
DEFAULT_SYSTEM_MESSAGE = "Hello! I'm your AI assistant. How can I help you today?"

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="🤖 AI Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-stats {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = 0
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

def save_conversation_to_file():
    """Save current conversation to JSON file"""
    try:
        filename = f"conversation_{st.session_state.conversation_id}.json"
        conversation_data = {
            "conversation_id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.chat_history,
            "stats": {
                "total_messages": st.session_state.total_messages,
                "error_count": st.session_state.error_count
            }
        }
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        return filename
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        return None

def load_conversation_from_file(uploaded_file):
    """Load conversation from uploaded JSON file"""
    try:
        conversation_data = json.load(uploaded_file)
        st.session_state.chat_history = conversation_data.get("messages", [])
        st.session_state.total_messages = conversation_data.get("stats", {}).get("total_messages", 0)
        st.session_state.error_count = conversation_data.get("stats", {}).get("error_count", 0)
        return True
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return False

def trim_chat_history():
    """Keep chat history within reasonable limits"""
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

def typing_effect(text: str, container, speed: float = TYPING_SPEED) -> None:
    """Create typing effect for assistant responses"""
    full_response = ""
    words = text.split()
    
    for i, word in enumerate(words):
        full_response += word + " "
        container.write(full_response + "▌")
        time.sleep(speed)
    
    container.write(full_response.strip())

# ---- Initialize Session State ----
initialize_session_state()

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Settings & Controls")
    
    # Agent Configuration
    st.subheader("🔧 Agent Config")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Response Length", 100, 2000, 500, 100)
    
    # Chat Management
    st.subheader("💬 Chat Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_messages = 0
            st.session_state.error_count = 0
            st.rerun()
    
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.total_messages = 0
            st.session_state.error_count = 0
            st.rerun()
    
    # File Operations
    st.subheader("📁 File Operations")
    
    # Save conversation
    if st.button("💾 Save Conversation", use_container_width=True):
        if st.session_state.chat_history:
            filename = save_conversation_to_file()
            if filename:
                st.success(f"Saved as {filename}")
            else:
                st.error("Failed to save conversation")
        else:
            st.warning("No conversation to save")
    
    # Load conversation
    uploaded_file = st.file_uploader("📤 Load Conversation", type=['json'])
    if uploaded_file is not None:
        if load_conversation_from_file(uploaded_file):
            st.success("Conversation loaded successfully!")
            st.rerun()
        else:
            st.error("Failed to load conversation")
    
    # Chat Statistics
    st.subheader("📊 Chat Statistics")
    st.markdown(f"""
    <div class="chat-stats">
        <strong>Session ID:</strong> {st.session_state.conversation_id}<br>
        <strong>Total Messages:</strong> {st.session_state.total_messages}<br>
        <strong>Current History:</strong> {len(st.session_state.chat_history)} messages<br>
        <strong>Errors:</strong> {st.session_state.error_count}
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        enable_typing_effect = st.checkbox("Enable Typing Effect", value=True)
        typing_speed = st.slider("Typing Speed", 0.01, 0.1, TYPING_SPEED, 0.01)
        show_timestamps = st.checkbox("Show Message Timestamps", value=False)
        debug_mode = st.checkbox("Debug Mode", value=False)

# ---- Main Chat Interface ----
st.title("💡 AI Agent Assistant")


# Show welcome message if no chat history
if not st.session_state.chat_history:
    st.info("👋 " + DEFAULT_SYSTEM_MESSAGE)

# ---- Display Chat History ----
chat_container = st.container()

with chat_container:
    for i, (role, message, *extra) in enumerate(st.session_state.chat_history):
        timestamp = extra[0] if extra else None
        
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                if show_timestamps and timestamp:
                    st.caption(f"*{timestamp}*")
                st.write(message)
        
        elif role == "assistant":
            with st.chat_message("assistant", avatar="💡"):
                if show_timestamps and timestamp:
                    st.caption(f"*{timestamp}*")
                st.write(message)
        
        elif role == "error":
            st.markdown(f"""
            <div class="error-message">
                <strong>⚠️ Error:</strong> {message}
            </div>
            """, unsafe_allow_html=True)

# ---- Chat Input ----
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message with timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input, current_time))
    st.session_state.total_messages += 1
    
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        if show_timestamps:
            st.caption(f"*{current_time}*")
        st.write(user_input)
    
    # Get assistant response
    with st.chat_message("assistant", avatar="💡"):
        response_container = st.empty()
        
        try:
            with st.spinner("🤔 Thinking..."):
                # Call your agent with additional parameters
                response = ask_agent(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    conversation_history=st.session_state.chat_history if hasattr(ask_agent, '__code__') and 'conversation_history' in ask_agent.__code__.co_varnames else None
                )
            
            # Display response with typing effect
            response_time = datetime.now().strftime("%H:%M:%S")
            if show_timestamps:
                st.caption(f"*{response_time}*")
            
            if enable_typing_effect:
                typing_effect(response, response_container, typing_speed)
            else:
                response_container.write(response)
            
            # Save assistant response
            st.session_state.chat_history.append(("assistant", response, response_time))
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.error_count += 1
            
            if debug_mode:
                st.exception(e)
            else:
                response_container.markdown(f"""
                <div class="error-message">
                    <strong>⚠️ Error:</strong> {error_msg}
                </div>
                """, unsafe_allow_html=True)
            
            # Log error
            logger.error(f"Agent error: {e}")
            st.session_state.chat_history.append(("error", error_msg, datetime.now().strftime("%H:%M:%S")))
    
    # Trim chat history to prevent memory issues
    trim_chat_history()
    
    # Auto-save periodically
    if len(st.session_state.chat_history) % 10 == 0:
        save_conversation_to_file()

