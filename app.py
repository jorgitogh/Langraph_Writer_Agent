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

                  NOTE: Do not write the article.
                  Use Tavily search if needed, but call tools at most once.
                  Then return a plain-text summary of the key findings for the outliner node.
                  """

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline
                       for the article.
                    """

writer_template = """Your job is to write an article, do it in this format:

                        TITLE: <title>
                        BODY: <body>

                      NOTE: Do not copy the outline. You need to write the article with the info provided by the outline.
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
    if getattr(last_message, "tool_calls", None):
        return "tools"
    # Otherwise, we stop (send state to outliner)
    return "outliner"


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


@st.cache_resource(show_spinner=False)
def build_graph(gemini_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
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
    st.write("Provide your Gemini and Tavily API keys, then request an article.")

    gemini_api_key = st.text_input("Gemini API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    user_prompt = st.text_area("What article should I write for you?")

    if st.button("Generate Article"):
        if not gemini_api_key.strip() or not tavily_api_key.strip():
            st.error("Please fill Gemini API key and Tavily API key.")
            return
        if not user_prompt.strip():
            st.error("Please enter an article request.")
            return

        try:
            with st.spinner("Generating article..."):
                graph = build_graph(gemini_api_key.strip(), tavily_api_key.strip())
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_prompt.strip())]},
                    config={"recursion_limit": 50},
                )

            messages = result.get("messages", [])
            output_text = ""
            for msg in reversed(messages):
                output_text = message_text(msg)
                if output_text:
                    break

            if output_text:
                st.subheader("Generated Article")
                st.write(output_text)
            else:
                st.error("The model returned an empty response. Try again with a more specific prompt.")
                with st.expander("Debug info"):
                    st.write(messages)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
