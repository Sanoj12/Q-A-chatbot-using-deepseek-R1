import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.vectorstores import InMemoryVectorStore

from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import OllamaLLM



##CREATE PROPMT TEMPLATE

PROMPT_TEMPLATE =  """
   you are an expert research assistant use the provide context to answer the query
    
   Query:{user_query}
   Context: {document_context}
"""


#path define to store pdf

PDF_STORAGE_PATH = 'document_store/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1-1.5b")


def save_uploaded_file(uploaded_file):
    file_path =PDF_STORAGE_PATH + uploaded_file.name

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return file_path


def load_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()



###chunks document


def chunk_document(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index=True
    )

    return text_processor.split_documents(raw_documents)

 #after chunkking data convert into embedding and store it into vector db


def index_documents(documents_chunks):
    DOCUMENT_VECTOR_DB.add_documents(documents_chunks)


###cosine similary


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


def generate_answer(user_query,document_context):
    context_text = "\n\n".join([doc.page_content for doc in document_context])

    conversation_prompt =ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    response_chain = conversation_prompt | LANGUAGE_MODEL

    return response_chain.invoke({"user_query":user_query, "document_context":context_text})


####UI


st.title("Sample Document Q&A AI")
st.markdown("Document Assistant")


#file upload selection

uploaded_pdf =  st.file_uploader(
    "Upload Document File",
     type="pdf",
     help="Select a pdf for analysis",
     accept_multiple_files=False
)



##execute file

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_doc = load_documents(saved_path)
    process_chunks = chunk_document(raw_doc)
    index_documents(process_chunks)

    st.success("Document uploaded and processed successfully!")

    user_input = st.text_input("enter your question about the document")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        
        with st.spinner("Loading..... please wait..."):
            relevant_docs = find_related_documents(user_input)

            al_response = generate_answer(user_input, relevant_docs) 

        
        with st.chat_message("assistant"):
            st.write(al_response)