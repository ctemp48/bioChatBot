from main import ChatBot
import streamlit as st
import uuid
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

bot = ChatBot()
if 'id' not in st.session_state:
    st.session_state['id'] = str(uuid.uuid4())

st.set_page_config(page_title="Ask About Christian Templin")

st.title('Ask About Christian Templin')

def generate_response(input):
    result = bot.conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": st.session_state['id']}
        },  
    )
    return result["answer"]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about Christian and I will "
                                  "try to answer!"}]
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})