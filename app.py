import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import PyPDF2
import io
import os

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def process_pdf(pdf_file):
    # 直接從上傳的文件讀取 PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
    
    # 提取文本
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # 分割文本
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    
    # 創建文檔
    documents = [Document(page_content=t) for t in texts]
    
    # 創建向量存儲
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 創建檢索鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

st.title("PDF 智能問答系統 (使用 GPT-4)")

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
            st.write(f"內容: {doc.page_content[:200]}...")

st.sidebar.write("本應用程式使用 Langchain、FAISS 向量數據庫和 GPT-4 模型進行智能問答。")
st.sidebar.info("注意：本應用程式需要 OpenAI API 金鑰才能運作。")
