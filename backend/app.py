from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from langchain_core.messages import ToolMessage, HumanMessage, AIMessageChunk
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_tool = TavilySearchResults(max_results=4)
tools = [search_tool]

memory = MemorySaver()

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result],
    }


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result],
    }


async def tools_router(state: State):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return "end"


async def tool_node(state: State):
    """Custom tool node that handles tool calls from the LLM"""
    tool_calls = state["messages"][-1].tool_calls

    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            search_results = await search_tool.ainvoke(tool_args)

            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name,
            )

            tool_messages.append(tool_message)

    return {"messages": tool_messages}


graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges(
    "model", tools_router, {"tool_node": "tool_node", "end": END}
)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )


async def generate_chat_response(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())

        config = {"configurable": {"thread_id": new_checkpoint_id}}

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config,
        )

        yield f'data : {{"type":"checkpoint","checkpoint_id":"{new_checkpoint_id}"}}\n\n'
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config,
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_message = serialise_ai_message_chunk(event["data"]["chunk"])

            safe_content = chunk_message.replace("'", "\\").replace("\n", "\\n")
            yield f'data: {{"type" : "content","content":"{safe_content}"}}\n\n'

        elif event_type == "on_chat_model_end":
            tool_calls = (
                event["data"]["output"].tool_calls
                if hasattr(event["data"]["output"], "tool_calls")
                else []
            )
            search_calls = [
                call
                for call in tool_calls
                if call["name"] == "tavily_search_results_json"
            ]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = (
                    search_query.replace('"', "\\")
                    .replace("'", "\\")
                    .replace("\n", "\\n")
                )

                yield f'data: {{"type" : "search_start","query":"{safe_query}"}}\n\n'

        elif (
            event_type == "on_tool_end"
            and event["name"] == "tavily_search_results_json"
        ):
            output = event["data"]["output"]

            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])

                urls_json = json.dumps(urls)
                yield f'data: {{"type" : "search_results","urls":{urls_json}}}\n\n'

    yield f'data : {{"type" : "end"}}\n\n'


@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = None):
    return StreamingResponse(
        generate_chat_response(message, checkpoint_id),
        media_type="text/event-stream",
    )
