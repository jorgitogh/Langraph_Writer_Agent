import functools
import os
from typing import Annotated, Literal, TypedDict

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_agent(llm, tools, system_message: str):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm
    

search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.

                  NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node.
                  """

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline
                       for the article.
                    """

writer_template = """Your job is to write an article, do it in this format:

                        TITLE: <title>
                        BODY: <body>

                      NOTE: Do not copy the outline. You need to write the article with the info provided by the outline.

                       ```
                    """
            
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return{
        'messages':[result]
    }
    
def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (send state to outliner)
    return "outliner"


@st.cache_resource(show_spinner=False)
def build_graph(model_name: str, gemini_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_api_key)
    tools = [TavilySearchResults(max_results=5)]

    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


def main():
    st.set_page_config(page_title="LangGraph Writer Agent")
    st.title("LangGraph Writer Agent")
    st.write("Provide your Gemini and Tavily settings, then request an article.")

    gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    user_prompt = st.text_area("What article should I write for you?")
    gemini_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if st.button("Generate Article"):
        if not gemini_model.strip() or not tavily_api_key.strip():
            st.error("Please fill Gemini model and Tavily API key.")
            return
        if not gemini_api_key:
            st.error("Set your Gemini key in environment variable GOOGLE_API_KEY.")
            return
        if not user_prompt.strip():
            st.error("Please enter an article request.")
            return

        try:
            graph = build_graph(gemini_model.strip(), gemini_api_key.strip(), tavily_api_key.strip())
            result = graph.invoke({"messages": [HumanMessage(content=user_prompt.strip())]})
            final_message = result["messages"][-1]
            st.subheader("Generated Article")
            st.write(final_message.content)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
