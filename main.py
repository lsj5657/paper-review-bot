from agents.url_search_agent import initialize_url_search_agent



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

    url = get_paper_url(title)
    print("URL: ", url)
    print(f"Download and save the paper to the '{title}.pdf'")


