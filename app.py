import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import tempfile
import os

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def process_pdf(pdf_file):
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # 載入和分割文檔
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 創建向量存儲
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 創建檢索鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # 刪除臨時文件
    os.unlink(tmp_file_path)

    return qa_chain

st.title("PDF 智能問答系統 (使用 gpt-4-mini)")

uploaded_file = st.file_uploader("請選擇一個PDF檔案", type="pdf")

if uploaded_file is not None:
    qa_chain = process_pdf(uploaded_file)
    st.success("PDF上傳並處理成功！")

    user_question = st.text_input("請輸入您的問題：")

    if user_question:
        with st.spinner("正在生成答案..."):
            result = qa_chain({"query": user_question})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.subheader("答案：")
        st.write(answer)

        st.subheader("參考來源：")
        for i, doc in enumerate(source_docs):
            st.write(f"來源 {i+1}:")
            st.write(f"頁碼: {doc.metadata['page']}")
            st.write(f"內容: {doc.page_content[:200]}...")
