from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

class Chatbot():
    load_dotenv()
    loader = TextLoader('./biography.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(docs, embedding_function)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    llm = HuggingFaceEndpoint(repo_id=repo_id, 
                        temperature=0.1,
                        top_k=50, 
                        huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN'))

    template = """
    You are answering questions about a person named Christian. Use the provided 
    context to answer these questions. If the context does not provide an answer, just 
    say that you do not know. Do not answer any questions that are not relevant to Christian.

    Context: {context}
    Question: {question}
    Answer:

    """

    prompt = PromptTemplate(template = template, input_variables=["context", "question"])

    rag_chain = (
    {"context": db.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
    )

bot = Chatbot()
input=input("Ask me anything: ")
result = bot.rag_chain.invoke(input)
print(result)