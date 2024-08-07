from main import Chatbot
import streamlit as st
import uuid


bot = Chatbot()
session_id = uuid.uuid4

st.set_page_config(page_title="Ask About Christian Templin")
with st.sidebar:
    st.title('Ask About Christian Templin')

def generate_response(input):
    result = bot.conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": str(session_id)}
        },  
    )
    return result["answer"]

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about Christian and I will"
                                  "try to answer!"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Answering your question..."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)