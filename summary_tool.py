from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

def summarize_paper(pdf_path: str) -> str:
    """
    Summarizes the paper text.
    """

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["paper_text"],
        template="Summarize the following paper text: {paper_text}"
    )

    chain = prompt | llm


    result = chain.invoke({"paper_text": docs})
    return  result.content
    
