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
import requests
import re

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 初始化 session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# 預設的 PDF 檔案 URL
PDF_URLS = {
    "學習新知": "https://drive.google.com/file/d/1yhJvKTfaG_uSWJQv3oAVs__03_IZCGLX/view?usp=sharing",
    "課程問答": "https://drive.google.com/file/d/11DG5SOJb7nmlpcqcA2d2Vb_PoFltacn2/view?usp=sharing",
    "股市早報": "https://drive.google.com/file/d/14cmJF9-wnRYgDbMd8DRQS4KdhyO2axjN/view"
}


# 定義按鈕樣式的 CSS
button_style = """
<style>
.custom-button {
    display: inline-block;
    padding: 0.5em 1em;
    color: #FFFFFF;
    background-color: #80CBC4;
    border-radius: 5px;
    text-decoration: none;
    font-weight: bold;
    text-align: center;
    transition: background-color 0.3s;
}
.custom-button:hover {
    background-color: #45a049;
}
</style>
"""

# 添加 CSS 到頁面
st.markdown(button_style, unsafe_allow_html=True)

# 創建一個看起來像按鈕的鏈接
# st.markdown('<a href="https://lunarexp.streamlit.app/" target="_blank" class="custom-button">前往查看農民曆</a>', unsafe_allow_html=True)


# 其餘代碼保持不變
def get_pdf_from_google_drive(url):
    file_id = re.findall(r'/file/d/([^/]+)', url)[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    return io.BytesIO(response.content)

def get_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"讀取 PDF 時發生錯誤：{str(e)}")
        return None

def process_pdf(pdf_file):
    text = get_pdf_text(pdf_file)
    if text is None:
        return None, None
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
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
    根據以下文件摘要，生成5個相關的問題：

    摘要：{summary}

    請用繁體中文提供5個與此摘要相關的問題，每個問題獨立成行：
    """
    prompt = PromptTemplate(template=template, input_variables=["summary"])
    questions = llm.predict(prompt.format(summary=summary))
    return questions.strip().split('\n')

st.title("PDF 智能問答系統 (PDF Summary & QA)")

# 選擇 PDF 來源
pdf_source = st.radio("選擇 PDF 來源(PDF Source)", ["學習新知-New Knowledge from Cloud Drives", "課程問答-Course QA from Study Group", "股市早報-Stock infos from Service Agent", "自訂上傳檔案-Upload your file"])

if pdf_source in ["學習新知-New Knowledge from Cloud Drives", "課程問答-Course QA from Study Group", "股市早報-Stock infos from Service Agent"]:
    pdf_url = PDF_URLS[pdf_source]
    if 'drive.google.com' in pdf_url:
        pdf_file = get_pdf_from_google_drive(pdf_url)
    else:
        response = requests.get(pdf_url)
        pdf_file = io.BytesIO(response.content)
elif pdf_source == "自訂上傳檔案(Upload your file)":
    uploaded_file = st.file_uploader("請選擇一個PDF檔案(Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_file = uploaded_file
    else:
        st.warning("請上傳一個 PDF 檔案(Please upload a PDF")
        st.stop()

# 添加確認按鈕
if st.button("確認選擇並處理 PDF (Comfirmation"):
    if 'pdf_file' in locals():
        with st.spinner("正在處理PDF文件(Processing the PDF)..."):
            st.session_state.qa_chain, documents = process_pdf(pdf_file)
            if st.session_state.qa_chain is not None:
                st.session_state.summary = get_summary(documents)
                st.session_state.questions = generate_questions(st.session_state.summary)
                st.session_state.pdf_processed = True
                st.success("PDF處理成功(Success)！")
            else:
                st.error("PDF 處理失敗，請檢查文件是否有效。")
    else:
        st.error("請先選擇或上傳一個 PDF 文件。")

if st.session_state.pdf_processed:
    st.subheader("文件摘要(Summary)")
    st.write(st.session_state.summary)

    st.subheader("選擇或輸入問題(Questions?)")
    question_options = ["請選擇一個問題"] + st.session_state.questions + ["自定義問題"]
    selected_question = st.selectbox("", question_options, key="question_select")

    if selected_question == "自定義問題":
        user_question = st.text_input("請輸入您的問題：", key="custom_question")
    elif selected_question != "請選擇一個問題":
        user_question = selected_question
    else:
        user_question = ""

    if user_question:
        if st.button("生成答案(Generating Answer"):
            with st.spinner("正在生成答案(Generating Answr..."):
                result = st.session_state.qa_chain({"query": user_question})
                answer = result["result"]
                source_docs = result["source_documents"]

            st.subheader("答案(Answer)：")
            st.write(answer)

            st.subheader("參考來源(Reference)：")
            for i, doc in enumerate(source_docs):
                st.write(f"來源 {i+1}:")
                st.write(f"內容: {doc.page_content[:200]}...")

#st.write("本應用程式使用 Langchain、FAISS 向量數據庫和 GPT-4 模型進行智能問答。")
#st.info("注意：本應用程式需要 OpenAI API 金鑰才能運作。")
