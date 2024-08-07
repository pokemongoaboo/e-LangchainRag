import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import PyPDF2
import io
import os

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource
def process_pdf(pdf_file):
    text = get_pdf_text(pdf_file)
    
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

    return qa_chain, documents

def get_summary(documents):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt_template = """請用繁體中文總結以下文件的內容，並提供一個簡潔的摘要：

    {text}

    繁體中文摘要："""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    summary = chain.run(documents)
    return summary

def generate_questions(summary):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    template = """
    根據以下文件摘要，生成3-5個相關的問題：

    摘要：{summary}

    請用繁體中文提供3-5個與此摘要相關的問題：
    """
    prompt = PromptTemplate(template=template, input_variables=["summary"])
    questions = llm.predict(prompt.format(summary=summary))
    return questions

st.title("PDF 智能問答系統 (使用 GPT-4)")

uploaded_file = st.file_uploader("請選擇一個PDF檔案", type="pdf")

if uploaded_file is not None:
    try:
        qa_chain, documents = process_pdf(uploaded_file)
        st.success("PDF上傳並處理成功！")

        with st.spinner("正在生成文件摘要..."):
            summary = get_summary(documents)
        
        st.subheader("文件摘要")
        st.write(summary)

        with st.spinner("正在生成問題建議..."):
            questions = generate_questions(summary)
        
        st.subheader("建議的問題")
        st.write(questions)

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
    except Exception as e:
        st.error(f"處理過程中發生錯誤：{str(e)}")

st.sidebar.write("本應用程式使用 Langchain、FAISS 向量數據庫和 GPT-4 模型進行智能問答。")
st.sidebar.info("注意：本應用程式需要 OpenAI API 金鑰才能運作。")
