from agent_tools.search_tool import get_paper_url
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.react.agent import create_react_agent
from langchain.agents import Tool, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def initialize_url_search_agent():
    """
    Initializes and returns the ReAct agent and its executor for searching paper URLs.
    """
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Tool 정의
    tools = [
        Tool(
            name="SearchPaperURL",
            func=get_paper_url,
            description="Given a paper title, find the URL of the paper."
        )
    ]

    # Prompt 템플릿 정의
    template = """
    You are provided with the {title} of a research paper. Your task is to find and provide the URL of the paper.

    Please use the tools available to you to find the URL of the paper.

    If you are unable to find the URL, return the string 'URL not found'.
    """
    
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["title"]
    )

    # React Prompt 로드
    react_prompt = hub.pull("hwchase17/react")
    if react_prompt is None:
        raise ValueError("Failed to load React prompt from the hub.")

    # Agent 생성
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=react_prompt
    )

    # Agent Executor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor, prompt_template
