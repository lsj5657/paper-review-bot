from summary_tool import load_document, save_to_vector_db, summarize_paper
from agents.url_search_agent import initialize_url_search_agent

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain

from langchain import hub

def initialize_rag_agent(index_name: str):
    """
    Initializes the RAG agent with Pinecone and embedding model.

    Args:
        index_name (str): The name of the Pinecone index.
        embedding_model (str): The embedding model to use (e.g., 'text-embedding-ada-002').

    Returns:
        Pinecone: A Pinecone vector store instance.
    """
    # Embedding 모델 초기화
    embeddings = OpenAIEmbeddings()

    vectorstore = Pinecone.from_existing_index(index_name =index_name, embedding=embeddings)

    return vectorstore


def answer_question_with_rag(vectorstore, question: str):
    """
    Answers a question using Retrieval-Augmented Generation (RAG).

    Args:
        vectorstore (Pinecone): The Pinecone vector store instance.
        question (str): The question to answer.
        llm_model (str): The LLM model to use for answering the question.

    Returns:
        str: The generated answer.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retreival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retreival_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        combine_docs_chain = combine_docs_chain,
        retriever = vectorstore.as_retriever()
        )

    result = retrieval_chain.invoke({"input": question})

    return result['answer']



def get_paper_url(title: str) -> str:

    # URL Search Agent 초기화
    agent_executor, prompt_template = initialize_url_search_agent()

    # 프롬프트 템플릿을 사용하여 입력값 생성
    formatted_input = prompt_template.format(title=title)


    result = agent_executor.invoke({"input": formatted_input})
    return result['output']



if __name__ == "__main__":
    # 사용자 입력
    title = input("Enter the paper title: ")


    print("------- Searching for the paper URL -------")

    url = get_paper_url(title)
    print("URL: ", url)
    print(f"Download the paper and save it to the 'papers' folder with the title '{title}.pdf'")

    pdf_path = f"papers/{title}.pdf"
    docs = load_document(pdf_path)

    
    print("------- Summarizing the paper -------")
    summary = summarize_paper(docs)
    print(summary)

    print("------- Storing the paper in Pinecone Vector DB -------")
    #save_to_vector_db(docs, "paper-index")


    print("------- Initializing RAG Agent -------")
    vectorstore = initialize_rag_agent(index_name="paper-index", embedding_model="text-embedding-ada-002")


    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        print("------- Generating Answer -------")
        answer = answer_question_with_rag(vectorstore, question)
        print("Answer:", answer)



