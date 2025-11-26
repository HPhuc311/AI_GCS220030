import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tools import ALL_TOOLS, get_data_statistics, predict_survival
import os

# --- UTILITY FUNCTIONS FOR APPLICATION STABILITY ---

TOOL_MAP = {
    "get_data_statistics": get_data_statistics,
    "predict_survival": predict_survival,
}

SYSTEM_INSTRUCTION = (
    "You are a professional and friendly AI assistant for a manager. "
    "Your task is to answer questions, perform data analysis, and "
    "provide predictions when requested. Always use the provided tools (functions) when necessary. "
    "Your dataset is about Titanic passengers. Provide your answers in natural language, do not display JSON code."
)

# 1. Cache Client and Chat Session (FIXED: Fixed UnhashableParamError)
@st.cache_resource
def get_gemini_client_and_chat(system_instruction, _tools):
    """
    Creates and caches both the Client and the Chat Session. The _tools parameter is prefixed 
    with an underscore so Streamlit does not hash it (fixing the UnhashableParamError).
    """
    try:
        # The Client will automatically read GEMINI_API_KEY from the environment variable
        client = genai.Client(api_key="AIzaSyBm6C1mShT2NImNPVkXxezsJMjyrqOq_gI")
        
        # Configure the model and start the chat session
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=_tools,
        )
        
        # Set a default model if a specific one is not available, using a robust model for RAG/Tool Use
        model_name = "gemini-2.5-flash"

        chat = client.chats.create(
            model=model_name,
            config=config,
            history=[], # Start with an empty history
        )
        return client, chat
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.stop()
        
# --- MAIN APPLICATION LOGIC ---

# 2. Initialize the Client and Chat Session
client, chat = get_gemini_client_and_chat(SYSTEM_INSTRUCTION, ALL_TOOLS)

# 3. Streamlit UI Configuration
st.set_page_config(page_title="Titanic Data Assistant", layout="centered")
st.title("🚢 Titanic Data Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask a question about the Titanic data..."):
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get assistant's response in a new message block
    with st.chat_message("assistant"):
        # Use a placeholder to temporarily display "Thinking..."
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Send user message to the LLM
            response = chat.send_message(prompt)
            
            # Function/Tool call loop
            while response.function_calls:
                # Update the placeholder to show function call status
                message_placeholder.text("  > Function call requested...") 

                tool_results = []
                for f_call in response.function_calls:
                    function_name = f_call.name
                    args = dict(f_call.args)
                    
                    # Update the placeholder to show which function is running
                    message_placeholder.text(f"  > Calling function: {function_name} with args: {args}")
                    
                    if function_name in TOOL_MAP:
                        function_to_call = TOOL_MAP[function_name]
                        function_result = function_to_call(**args)
                        
                        tool_results.append(types.Part.from_function_response(
                            name=function_name,
                            response={"result": function_result},
                        ))
                    else:
                        st.error(f"Error: Function {function_name} not found")
                        
                # Send the function execution results back to the LLM
                response = chat.send_message(tool_results)

            # 3. Display the final LLM response
            assistant_response = response.text
            
            # FIX: Update the placeholder with the final response text
            message_placeholder.markdown(assistant_response) 
            
            # Append to history *after* it's rendered successfully by the placeholder
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            

        except APIError as e:
            message_placeholder.error(f"An API error occurred: Please check your API Key and usage limits. Details: {e}")
            # Do NOT append error messages to history if they shouldn't be re-displayed on every rerun
        except Exception as e:
            message_placeholder.error(f"An unknown error occurred: {e}")
            # Do NOT append error messages to history if they shouldn't be re-displayed on every rerun