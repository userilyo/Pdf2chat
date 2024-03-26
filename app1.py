from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import logging

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=4036,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            st.error("An error occurred while processing your request.")

if __name__ == '__main__':
    main()
