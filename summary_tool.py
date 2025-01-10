from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# Pinecone 초기화
def load_document(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def save_to_vector_db(docs, index_name: str):
    # 문서 Split
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Pinecone Vector DB 저장
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="paper-index",
    )

    
def summarize_paper(docs: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["paper_text"],
        template="Summarize the following paper text: {paper_text}"
    )
    chain = prompt | llm
    result = chain.invoke({"paper_text": docs})
    return result.content


