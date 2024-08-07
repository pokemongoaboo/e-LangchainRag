import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import tempfile
import os

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def process_pdf(pdf_file, model_name):
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
        llm=ChatOpenAI(model_name=model_name, temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # 刪除臨時文件
    os.unlink(tmp_file_path)

    return qa_chain

st.title("增強版 PDF 智能問答系統")

# 模型選擇
model_options = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "GPT-4o mini": "gpt-4o-mini"
}
selected_model = st.sidebar.selectbox("選擇 OpenAI 模型", list(model_options.keys()))
model_name = model_options[selected_model]

uploaded_file = st.file_uploader("請選擇一個PDF檔案", type="pdf")

if uploaded_file is not None:
    qa_chain = process_pdf(uploaded_file, model_name)
    st.success("PDF上傳並處理成功！")

    user_question = st.text_input("請輸入您的問題：")

    if user_question:
        with st.spinner(f"使用 {selected_model} 生成答案..."):
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

st.sidebar.write(f"當前使用的模型: {selected_model}")
st.sidebar.write("本應用程式使用 Langchain 和 FAISS 向量數據庫進行智能問答。")
st.sidebar.info("注意：本應用程式需要 OpenAI API 金鑰才能運作。")
