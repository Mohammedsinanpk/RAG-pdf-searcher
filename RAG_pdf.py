import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
st.title("RAG searcher")
que= st.text_input("Ask about corona cases till 11 october 2020...")
button = st.button("search")

gpt_api_key = os.environ.get('GPT_API_key')
hf_key = os.environ.get('HF_key')
loader = PyPDFLoader("corona.pdf")
pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter())
vector_store=FAISS.from_documents(documents=pages,embedding=OpenAIEmbeddings(openai_api_key=gpt_api_key))
retriever=vector_store.as_retriever(search_kwargs={"k":5})
prompt = PromptTemplate(
    input_variables=['question','context'],
    template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer very much concise.Just answer the question dont give any further info. Question: {question} Context: {context} Answer:"
)
llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=gpt_api_key)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
ragchain=(
    {"context":retriever | format_docs,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
if button:
    st.text(ragchain.invoke(que))
