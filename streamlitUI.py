from main import ChatBot
import streamlit as st
import uuid

#create chatbot object
bot = ChatBot()

#create session id to track conversation history
if 'id' not in st.session_state:
    st.session_state['id'] = str(uuid.uuid4())

st.set_page_config(page_title="Ask About Christian Templin")

st.title('Ask About Christian Templin')

#function to generate response from llm
def generate_response(input):
    result = bot.conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": st.session_state['id']}
        },  
    )
    return result["answer"]

#create intro message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about Christian and I will "
                                  "try to answer!"}]
 
#write each message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# accept user input and pass as prompt to llm
if prompt := st.chat_input("Enter a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})