import os
from typing import Annotated, Sequence, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from openai import OpenAI
import uuid
from datetime import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.types import Command

import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
from contextlib import asynccontextmanager


load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = None


# Define a simple last-value reducer function
def get_last_value(existing_value: Any, new_value: Any) -> Any:
    """Return the new value, ignoring the existing value."""
    return new_value


# Define our state
class EnhancedRAGState(TypedDict):
    conversation_id: Annotated[str, get_last_value]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: Annotated[str, get_last_value]
    tool_decision: Annotated[Dict[str, bool], get_last_value]
    rag_context: Annotated[str, get_last_value]
    search_results: Annotated[str, get_last_value]
    image_url: Annotated[str, get_last_value]
    response: Annotated[str, get_last_value]


# Define the available tools to use
class ToolDecision(BaseModel):
    use_rag: bool = Field(
        description="Whether to use RAG to retrieve past conversations"
    )
    use_search: bool = Field(
        description="Whether to use web search for current information"
    )
    use_image_gen: bool = Field(description="Whether to generate an image")


# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "conversation_history"


# Initialize vector store with Qdrant
async def setup_qdrant_vectorstore():
    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )

    return vectorstore


# Store conversation in vector database
async def store_conversation(
    conversation_id: str, human_message: str, ai_response: str
):
    """Store the conversation in the vector database"""
    timestamp = datetime.now().isoformat()

    # Format the conversation
    conversation_text = f"[{timestamp}] User: {human_message}\nAssistant: {ai_response}"

    # Create documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([conversation_text])

    # Add metadata
    for doc in docs:
        doc.metadata = {
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "type": "conversation",
        }

    # Add to vectorstore
    await vectorstore.aadd_documents(docs)

    return True


# 1. Tool Selection Node
async def select_tools(state: EnhancedRAGState) -> EnhancedRAGState:
    """Determine which tools to use based on the query"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are an AI assistant that determines which tools to use for a user query.
        You need to decide which of the following tools to use:
        
        1. RAG (Retrieval Augmented Generation): Use when the query references past conversations or previously discussed information
        2. Web Search: Use when the query asks about current events, facts that might need verification, or information you might not have
           - IMPORTANT: ALWAYS use web search for queries about weather, news, sports scores, stock prices, or any real-time information
           - ALWAYS use web search when users ask about "today" or current conditions
        3. Image Generation: Use when the user explicitly asks for an image or visual content to be created
        
        Respond with a JSON object with these keys: "use_rag", "use_search", "use_image_gen" with boolean values.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = prompt | llm | JsonOutputParser()
    decision = await chain.ainvoke({"query": state["query"]})

    state["tool_decision"] = decision
    if decision["use_search"]:
        return Command(
            goto="perform_web_search",
            update=state,
        )
    elif decision["use_image_gen"]:
        return Command(
            goto="generate_image",
            update=state,
        )
    elif decision["use_rag"]:
        return Command(
            goto="retrieve_from_rag",
            update=state,
        )
    else:
        return Command(
            goto="generate_response",
            update=state,
        )


# 2. RAG Retrieval Node
async def retrieve_from_rag(state: EnhancedRAGState) -> EnhancedRAGState:
    """Retrieve relevant context from vector database if needed"""
    # Retrieve documents
    docs = await vectorstore.asimilarity_search(
        state["query"],
        k=3,
        # not adding this because some frontend code is not updated to handle this
        # filter=models.Filter(
        #     must=[
        #         models.FieldCondition(
        #             key="conversation_id",
        #             match=models.MatchValue(value=state["conversation_id"]),
        #         )
        #     ]
        # ),
    )
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])
    state["rag_context"] = context

    return Command(
        goto="generate_response",
        update=state,
    )


# 3. DALL-E Image Generation Tool
@tool
def generate_dalle_image(prompt: str) -> str:
    """Generate an image using DALL-E"""
    try:
        client = OpenAI()
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="512x512",
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {str(e)}"


# Initialize tools
tavily_tool = TavilySearchResults(max_results=3)
dalle_tool = generate_dalle_image


# Web Search Node
async def perform_web_search(state: EnhancedRAGState) -> EnhancedRAGState:
    """Perform web search if needed"""
    # Execute the tavily search tool
    search_query = state["query"]
    search_results = await tavily_tool.ainvoke(search_query)

    # Convert results to string if needed
    if isinstance(search_results, list):
        formatted_results = []
        for result in search_results:
            formatted_results.append(f"Title: {result.get('title', 'No title')}")
            formatted_results.append(f"URL: {result.get('url', 'No URL')}")
            formatted_results.append(
                f"Content: {result.get('content', 'No content')[:300]}..."
            )
            formatted_results.append("")
        search_results = "\n".join(formatted_results)

    search_results = search_results.replace("{", "{{").replace("}", "}}")

    state["search_results"] = search_results
    return Command(
        goto="generate_response",
        update=state,
    )


# Image Generation Node
async def generate_image(state: EnhancedRAGState) -> EnhancedRAGState:
    """Generate an image if needed"""
    # Create an improved prompt for DALL-E
    image_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        Create a detailed and clear prompt for DALL-E to generate an image based on 
        the user's request. Focus on visual details, style, and composition.
        Keep it concise but descriptive.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = image_prompt_template | llm | StrOutputParser()
    enhanced_prompt = await chain.ainvoke({"query": state["query"]})

    # Execute the DALL-E tool
    image_url = await dalle_tool.ainvoke(enhanced_prompt)

    state["image_url"] = str(image_url)

    return Command(
        goto="generate_response",
        update=state,
    )


# Response Generation Node
async def generate_response(state: EnhancedRAGState) -> EnhancedRAGState:
    """Generate response based on all available information"""

    messages = [
        (
            "system",
            "You are a helpful AI assistant. Respond directly to the user's query in a clear and concise manner. "
            "If you have search results, use them to inform your response but don't mention the search process itself. "
            "If an image was generated, simply describe what's in the image and mention it's available for viewing.",
        )
    ]

    # Add context if available - but only if relevant
    if state["rag_context"] and state["rag_context"] != "":
        messages.append(
            ("system", "Context from past conversations:\n" + state["rag_context"])
        )

    if state["search_results"] and state["search_results"] != "":
        messages.append(("system", f"Web search results:\n {state['search_results']}"))

    if state["image_url"] and state["image_url"] != "":
        messages.append(
            (
                "system",
                "You have generated an image which is available at: "
                + state["image_url"]
                + ". Mention that an image has been created for the user and DO include the URL in your response.",
            )
        )

    # Add only the current query - don't include message history to avoid confusion
    messages.append(("user", state["query"]))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({})

    state["response"] = response
    return state


# Build the graph
def build_enhanced_graph():
    workflow = StateGraph(EnhancedRAGState)

    # Add nodes
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("retrieve_from_rag", retrieve_from_rag)
    workflow.add_node("perform_web_search", perform_web_search)
    workflow.add_node("generate_image", generate_image)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge("generate_response", END)

    # Set entry point
    workflow.set_entry_point("select_tools")

    return workflow.compile()


# Create the graph
enhanced_graph = build_enhanced_graph()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    vectorstore = await setup_qdrant_vectorstore()
    yield


# Create FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def serialize_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_enhanced_response(query: str, conversation_id: Optional[str] = None):
    """Generate response using the enhanced RAG system with streaming"""
    is_new_conversation = conversation_id is None

    if is_new_conversation:
        new_conversation_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": new_conversation_id}}

        # Initialize state with just the current query
        state = {
            "conversation_id": new_conversation_id,
            "messages": [],  # No message history for new conversations
            "query": query,
            "tool_decision": {
                "use_rag": False,
                "use_search": False,
                "use_image_gen": False,
            },
            "rag_context": "",
            "search_results": "",
            "image_url": "",
            "response": "",
        }

        # Send initial conversation ID
        yield f'data: {{"type":"conversation_start","conversation_id":"{new_conversation_id}"}}\n\n'
    else:
        config = {"configurable": {"thread_id": conversation_id}}

        # Add the current query only
        messages = []
        messages.append(HumanMessage(content=query))

        # Initialize state with messages
        state = {
            "conversation_id": conversation_id,
            "messages": messages,
            "query": query,
            "tool_decision": {
                "use_rag": False,
                "use_search": False,
                "use_image_gen": False,
            },
            "rag_context": "",
            "search_results": "",
            "image_url": "",
            "response": "",
        }

    # Stream graph execution events
    events = enhanced_graph.astream_events(state, config=config)

    async for event in events:
        event_type = event["event"]

        # Stream LLM content chunks
        if event_type == "on_chat_model_stream":
            chunk_message = serialize_ai_message_chunk(event["data"]["chunk"])
            safe_content = (
                chunk_message.replace("'", "\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )
            yield f'data: {{"type":"content","content":"{safe_content}"}}\n\n'

        # Tool usage events
        elif event_type == "on_chat_model_end":
            node_name = event.get("name", "")

            # Handle tool decision node
            if node_name == "select_tools":
                tool_decision = event["data"]["output"].get("tool_decision", {})

                if tool_decision.get("use_search", False):
                    safe_query = (
                        query.replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type":"search_start","query":"{safe_query}"}}\n\n'

                if tool_decision.get("use_image_gen", False):
                    safe_query = (
                        query.replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type":"image_gen_start","query":"{safe_query}"}}\n\n'

        # Web search results
        elif event_type == "on_node_end" and event.get("name") == "perform_web_search":
            search_results = event["data"]["output"].get("search_results", "")
            if search_results:
                # Extract URLs from search results
                urls = []
                for line in search_results.split("\n"):
                    if line.startswith("URL:"):
                        urls.append(line[4:].strip())

                urls_json = json.dumps(urls)
                yield f'data: {{"type":"search_results","urls":{urls_json}}}\n\n'

        # Image generation results
        elif event_type == "on_node_end" and event.get("name") == "generate_image":
            image_url = event["data"]["output"].get("image_url")
            if image_url and image_url != "":
                safe_url = image_url.replace('"', '\\"').replace("'", "\\'")
                yield f'data: {{"type":"image_generated","url":"{safe_url}"}}\n\n'
                print(f"Debug: Image generated and URL saved: {image_url}")

        # Response completion notification
        elif event_type == "on_node_end" and event.get("name") == "generate_response":
            output_data = event["data"]["output"]
            has_image = (
                bool(output_data.get("image_url"))
                and output_data.get("image_url") != ""
            )

            if has_image:
                safe_url = (
                    output_data["image_url"].replace('"', '\\"').replace("'", "\\'")
                )
                yield f'data: {{"type":"response_complete","has_image":true,"image_url":"{safe_url}"}}\n\n'
            else:
                yield f'data: {{"type":"response_complete","has_image":false}}\n\n'

    # Store conversation
    await store_conversation(
        conversation_id=state["conversation_id"],
        human_message=query,
        ai_response=state.get("response", "No response generated"),
    )

    # Signal the end of the stream
    yield f'data: {{"type":"end"}}\n\n'


@app.get("/rag_chat/{query}")
async def rag_chat_stream(query: str, conversation_id: Optional[str] = None):
    """Enhanced RAG chat endpoint with streaming response"""
    return StreamingResponse(
        generate_enhanced_response(query, conversation_id),
        media_type="text/event-stream",
    )


@app.get("/rag_chat/continue/{conversation_id}/{query}")
async def continue_rag_chat_stream(conversation_id: str, query: str):
    """Continue an existing RAG chat conversation"""
    return StreamingResponse(
        generate_enhanced_response(query, conversation_id),
        media_type="text/event-stream",
    )


# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "RAG Assistant API"}


# For direct execution of the script
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
