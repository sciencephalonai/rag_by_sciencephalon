# import PyPDF2
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# from pathlib import Path
import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

os.environ['OPENAI_API_KEY'] = st.secrets.openai_secret_key.openai_api_key

def main():
    st.set_page_config(page_title="RAG App by SciEncephalon AI", page_icon="📚") 
    st.image('website_logo.png', width=400)

    """
    SciEncephalon AI introduces a novel RAG application 📚, leveraging insights solely from "21st Century Marketing,
    "AI Tsunami Capability Statement," "AI Tsunami Amazon Book," and "Blockchain in Business." This tool synthesizes 
    data from these specific texts to offer strategic business and marketing insights, predicting trends and 
    formulating actionable intelligence. Designed to support but not replace strategic decision-making, 
    it aids in crafting tailored strategies and understanding market dynamics, emphasizing its utility 
    as a supplementary resource for innovation and growth in the digital era.
    """

    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)

    query = st.chat_input(placeholder="Ask me anything!")

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query:
        docs = new_db.similarity_search(query)
        chain = load_qa_chain(OpenAI(temperature=0), chain_type = 'stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question=query)
        with st.chat_message("user"):
            st.write(query)
            history.add_user_message(query)
        
        st.spinner(text="In progress...")

        with st.chat_message("assistant"):
            st.write(response)
            history.add_ai_message(response)
   
if __name__ == '__main__':
    main()
