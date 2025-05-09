import os
from typing import Annotated, Sequence, TypedDict, Dict, Any, Optional, List
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
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.types import Command
import httpx
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
from contextlib import asynccontextmanager
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


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
    song_lyrics: Annotated[str, get_last_value]
    song_audio_url: Annotated[str, get_last_value]
    song_audio_data: Annotated[Dict, get_last_value]
    video_url: Annotated[str, get_last_value]
    video_data: Annotated[Dict, get_last_value]
    response: Annotated[str, get_last_value]
    # Calendar-related fields
    calendar_action: Annotated[str, get_last_value]
    calendar_events: Annotated[List[Dict], get_last_value]
    calendar_free_busy: Annotated[List[Dict], get_last_value]
    calendar_result: Annotated[Dict, get_last_value]


# Define the available tools to use
class ToolDecision(BaseModel):
    use_rag: bool = Field(
        description="Whether to use RAG to retrieve past conversations"
    )
    use_search: bool = Field(
        description="Whether to use web search for current information"
    )
    use_image_gen: bool = Field(
        description="Whether to generate an image",
    )
    use_song_gen: bool = Field(
        description="Whether to generate a song or lyrics",
    )
    use_video_gen: bool = Field(
        description="Whether to generate a video",
    )
    use_calendar: bool = Field(
        description="Whether to use calendar functionality",
    )


# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "conversation_history"


# Google Calendar setup
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "Credentials.json"


def get_calendar_service():
    """Create and return a Google Calendar service object"""
    creds = None

    # Check if token file exists
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid credentials, get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    try:
        # Build and return the service
        service = build("calendar", "v3", credentials=creds)
        return service
    except Exception as e:
        print(f"Error building calendar service: {str(e)}")
        return None


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


# Add topic extraction functionality


async def extract_conversation_topic(conversation_text: str) -> str:
    """Extract topic from conversation text using LLM"""
    topic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Extract a short, concise topic (5 words or less) from this conversation.
                Focus on the main subject being discussed, not the format.
                Return only the topic, no explanations or additional text.
                """,
            ),
            ("user", "{conversation_text}"),
        ]
    )

    chain = topic_prompt | llm | StrOutputParser()
    topic = await chain.ainvoke({"conversation_text": conversation_text})

    return topic.strip()


# Modify store_conversation to extract and store topics
async def store_conversation(
    conversation_id: str,
    human_message: str,
    ai_response: str,
    calendar_event: Dict = None,
):
    """Store the conversation in the vector database with topic extraction and optional calendar event data"""
    timestamp = datetime.now().isoformat()
    date = datetime.now().strftime("%Y-%m-%d")  # Get current date for grouping

    # Format the conversation
    conversation_text = f"[{timestamp}] User: {human_message}\nAssistant: {ai_response}"

    # Extract topic if this is the first message in a conversation
    topic = ""
    try:
        # Check if this is the first message in the conversation
        existing_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(value=conversation_id),
                    ),
                ]
            ),
            limit=1,
        )

        is_first_message = len(existing_points[0]) == 0

        if is_first_message:
            # For the first message, extract a topic
            topic = await extract_conversation_topic(conversation_text)
    except Exception as e:
        print(f"Error checking for existing conversation or extracting topic: {str(e)}")

    # Create documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([conversation_text])

    # Add metadata - enhanced with date and topic
    for doc in docs:
        doc.metadata = {
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "date": date,  # Add date for daily grouping
            "type": "conversation",
            "topic": topic,  # Add topic if extracted
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
        4. Song Generation: Use when the user explicitly asks for a song, lyrics, poem, or musical content to be created
        5. Video Generation: Use when the user explicitly asks for a video, animation, or moving visual content to be created
        6. Calendar: Use when the user asks about scheduling, meetings, events, availability, or any calendar-related functionality
           - This includes creating/updating/deleting events, checking schedules, finding free time, or managing appointments
           
        Respond with a JSON object with these keys: "use_rag", "use_search", "use_image_gen", "use_song_gen", "use_video_gen", "use_calendar" with boolean values.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = prompt | llm | JsonOutputParser()
    decision = await chain.ainvoke({"query": state["query"]})

    state["tool_decision"] = decision
    if decision["use_calendar"]:
        return Command(
            goto="process_calendar_request",
            update=state,
        )
    elif decision["use_search"]:
        return Command(
            goto="perform_web_search",
            update=state,
        )
    elif decision["use_image_gen"]:
        return Command(
            goto="generate_image",
            update=state,
        )
    elif decision["use_song_gen"]:
        return Command(
            goto="generate_song",
            update=state,
        )
    elif decision["use_video_gen"]:
        return Command(
            goto="generate_video",
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
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(value=state["conversation_id"]),
                ),
            ]
        ),
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


# 4. Song Generation Tool
@tool
def generate_song_lyrics(prompt: str) -> str:
    """Generate song lyrics based on a prompt"""
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled songwriter. Create original song lyrics based on the given prompt. Include title, verses, and chorus.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating song lyrics: {str(e)}"


# 5. Video Generation Tool
@tool
def generate_video(prompt: str) -> Dict:
    """Generate video based on a prompt"""
    try:
        client = OpenAI()
        # This is a placeholder for actual video generation API
        # In a real implementation, you would call a video generation service like Runway ML, Replicate, etc.
        # For now, we'll simulate a response structure

        # For example, if using the OpenAI API for a future video generation feature
        # or using a third-party API through the OpenAI client

        video_api_key = os.getenv("VIDEO_API_KEY")
        video_api_url = os.getenv("VIDEO_API_URL")

        if not video_api_key or not video_api_url:
            return {
                "status": "error",
                "message": "Video API configuration is missing",
                "url": None,
            }

        # Simulating a video generation request
        # In a real implementation, you would make an HTTP request to the video generation API
        # For example:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         video_api_url,
        #         headers={"Authorization": f"Bearer {video_api_key}"},
        #         json={"prompt": prompt, "duration": 5}
        #     )
        #     result = response.json()

        # For demonstration purposes, return a mock response
        # In production, you would parse the actual API response
        # Mock a progress ID that could be used to poll for status
        progress_id = str(uuid.uuid4())

        return {
            "status": "processing",
            "id": progress_id,
            "message": "Video generation started",
            "estimated_time": "30 seconds",
            "url": f"https://example.com/video/{progress_id}",  # This would be a placeholder until processing completes
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating video: {str(e)}",
            "url": None,
        }


# 6. Calendar Tools
@tool
async def create_calendar_event(event_details: Dict) -> Dict:
    """
    Create a calendar event with the specified details.

    Parameters:
    - event_details (dict) with keys:
      - summary: Event title/name
      - description: Details about the event
      - start_time: ISO format datetime
      - end_time: ISO format datetime
      - attendees: Optional list of email addresses
      - location: Optional location
    """
    try:
        service = get_calendar_service()
        if not service:
            return {"status": "error", "message": "Could not access calendar service"}

        # Format attendees if any
        attendees = []
        if "attendees" in event_details and event_details["attendees"]:
            attendees = [
                {"email": email.strip()} for email in event_details["attendees"]
            ]

        # Create event object
        event = {
            "summary": event_details.get("summary", "New Event"),
            "location": event_details.get("location", ""),
            "description": event_details.get("description", ""),
            "start": {
                "dateTime": event_details.get("start_time"),
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": event_details.get("end_time"),
                "timeZone": "UTC",
            },
        }

        # Add attendees if present
        if attendees:
            event["attendees"] = attendees
            # Send notifications to attendees
            event["sendUpdates"] = "all"

        # Insert the event
        event = service.events().insert(calendarId="primary", body=event).execute()

        return {
            "status": "success",
            "message": "Event created successfully",
            "event_id": event.get("id"),
            "event_link": event.get("htmlLink"),
        }
    except Exception as e:
        return {"status": "error", "message": f"Error creating event: {str(e)}"}


@tool
async def check_calendar_availability(time_range: Dict) -> Dict:
    """
    Check calendar availability for a given time range.

    Parameters:
    - time_range (dict) with keys:
      - start_time: ISO format datetime
      - end_time: ISO format datetime
      - attendees: Optional list of email addresses to check availability for
    """
    try:
        service = get_calendar_service()
        if not service:
            return {"status": "error", "message": "Could not access calendar service"}

        # Set up time range
        start_time = time_range.get("start_time")
        end_time = time_range.get("end_time")

        # Prepare request body for freeBusy query
        body = {
            "timeMin": start_time,
            "timeMax": end_time,
            "items": [{"id": "primary"}],  # User's primary calendar
        }

        # Add attendees if provided
        attendees = time_range.get("attendees", [])
        if attendees:
            for attendee in attendees:
                if isinstance(attendee, str) and "@" in attendee:
                    body["items"].append({"id": attendee})

        # Query free/busy
        free_busy_request = service.freebusy().query(body=body)
        free_busy_response = free_busy_request.execute()

        # Process and format the response
        calendars = free_busy_response.get("calendars", {})
        busy_periods = {}

        for email, calendar_data in calendars.items():
            if "busy" in calendar_data:
                busy_periods[email] = calendar_data["busy"]

        # Determine if the requested time is available
        is_available = all(len(periods) == 0 for periods in busy_periods.values())

        # Find alternative times if not available
        alternatives = []
        if not is_available:
            # Create a list of potential 30-minute slots over the next 5 days
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            slot_duration = timedelta(minutes=30)
            num_days = 5

            # Generate slots for consideration
            current_slot = start_dt
            end_date = start_dt + timedelta(days=num_days)

            while current_slot < end_date:
                # Only consider business hours (9 AM to 5 PM)
                if (
                    9 <= current_slot.hour < 17 and current_slot.weekday() < 5
                ):  # Weekdays only
                    slot_end = current_slot + slot_duration

                    # Check if this slot conflicts with any busy periods
                    conflicts = False
                    for email, periods in busy_periods.items():
                        for period in periods:
                            period_start = datetime.fromisoformat(
                                period["start"].replace("Z", "+00:00")
                            )
                            period_end = datetime.fromisoformat(
                                period["end"].replace("Z", "+00:00")
                            )

                            # Check for overlap
                            if not (
                                slot_end <= period_start or current_slot >= period_end
                            ):
                                conflicts = True
                                break

                        if conflicts:
                            break

                    # If no conflicts, add to alternatives
                    if (
                        not conflicts and len(alternatives) < 5
                    ):  # Limit to 5 alternatives
                        alternatives.append(
                            {
                                "start": current_slot.isoformat() + "Z",
                                "end": slot_end.isoformat() + "Z",
                            }
                        )

                # Move to next slot
                current_slot += slot_duration

        return {
            "status": "success",
            "is_available": is_available,
            "busy_periods": busy_periods,
            "alternatives": alternatives,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking availability: {str(e)}"}


@tool
async def query_calendar_events(query_params: Dict) -> Dict:
    """
    Query calendar events based on date range, search term, or other parameters.

    Parameters:
    - query_params (dict) with keys:
      - start_time: ISO format datetime (optional)
      - end_time: ISO format datetime (optional)
      - max_results: Maximum number of events to return (optional, default 10)
      - search_term: Text to search for in events (optional)
    """
    try:
        service = get_calendar_service()
        if not service:
            return {"status": "error", "message": "Could not access calendar service"}

        # Set up parameters
        now = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
        start_time = query_params.get("start_time", now)
        end_time = query_params.get("end_time", None)
        max_results = query_params.get("max_results", 10)
        search_term = query_params.get("search_term", None)

        # Prepare query parameters
        params = {
            "calendarId": "primary",
            "timeMin": start_time,
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }

        # Add end time if provided
        if end_time:
            params["timeMax"] = end_time

        # Add search term if provided
        if search_term:
            params["q"] = search_term

        # Execute the query
        events_result = service.events().list(**params).execute()
        events = events_result.get("items", [])

        # Format events for response
        formatted_events = []
        for event in events:
            start = event.get("start", {})
            end = event.get("end", {})

            formatted_event = {
                "id": event.get("id"),
                "summary": event.get("summary", "Untitled Event"),
                "description": event.get("description", ""),
                "location": event.get("location", ""),
                "start": start.get("dateTime", start.get("date", "")),
                "end": end.get("dateTime", end.get("date", "")),
                "link": event.get("htmlLink", ""),
            }

            # Add attendees if present
            if "attendees" in event:
                formatted_event["attendees"] = [
                    {
                        "email": attendee.get("email", ""),
                        "name": attendee.get("displayName", ""),
                        "response_status": attendee.get("responseStatus", ""),
                    }
                    for attendee in event["attendees"]
                ]

            formatted_events.append(formatted_event)

        return {
            "status": "success",
            "events": formatted_events,
            "event_count": len(formatted_events),
        }
    except Exception as e:
        return {"status": "error", "message": f"Error querying calendar: {str(e)}"}


@tool
async def update_calendar_event(event_update: Dict) -> Dict:
    """
    Update an existing calendar event.

    Parameters:
    - event_update (dict) with keys:
      - event_id: ID of the event to update
      - summary: New event title (optional)
      - description: New event description (optional)
      - start_time: New start time (optional)
      - end_time: New end time (optional)
      - location: New location (optional)
      - attendees: New attendees list (optional)
    """
    try:
        service = get_calendar_service()
        if not service:
            return {"status": "error", "message": "Could not access calendar service"}

        # Get event ID
        event_id = event_update.get("event_id")
        if not event_id:
            return {"status": "error", "message": "Event ID is required"}

        # Get the existing event
        event = service.events().get(calendarId="primary", eventId=event_id).execute()

        # Update fields if provided
        if "summary" in event_update:
            event["summary"] = event_update["summary"]

        if "description" in event_update:
            event["description"] = event_update["description"]

        if "location" in event_update:
            event["location"] = event_update["location"]

        # Update start and end times if provided
        if "start_time" in event_update:
            event["start"]["dateTime"] = event_update["start_time"]

        if "end_time" in event_update:
            event["end"]["dateTime"] = event_update["end_time"]

        # Update attendees if provided
        if "attendees" in event_update and event_update["attendees"]:
            event["attendees"] = [
                {"email": email.strip()} for email in event_update["attendees"]
            ]
            # Set to notify attendees
            event["sendUpdates"] = "all"

        # Update the event
        updated_event = (
            service.events()
            .update(
                calendarId="primary", eventId=event_id, body=event, sendUpdates="all"
            )
            .execute()
        )

        return {
            "status": "success",
            "message": "Event updated successfully",
            "event_id": updated_event.get("id"),
            "event_link": updated_event.get("htmlLink"),
        }
    except Exception as e:
        return {"status": "error", "message": f"Error updating event: {str(e)}"}


@tool
async def delete_calendar_event(event_id_dict: Dict) -> Dict:
    """
    Delete a calendar event.

    Parameters:
    - event_id_dict (dict) with key:
      - event_id: ID of the event to delete
      - notify_attendees: Whether to notify attendees (boolean, optional)
    """
    try:
        service = get_calendar_service()
        if not service:
            return {"status": "error", "message": "Could not access calendar service"}

        # Get event ID
        event_id = event_id_dict.get("event_id")
        if not event_id:
            return {"status": "error", "message": "Event ID is required"}

        # Determine if attendees should be notified
        notify = "all" if event_id_dict.get("notify_attendees", True) else "none"

        # Delete the event
        service.events().delete(
            calendarId="primary", eventId=event_id, sendUpdates=notify
        ).execute()

        return {
            "status": "success",
            "message": "Event deleted successfully",
            "event_id": event_id,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error deleting event: {str(e)}"}


# Initialize tools
tavily_tool = TavilySearchResults(max_results=3)
dalle_tool = generate_dalle_image
song_tool = generate_song_lyrics
video_tool = generate_video
create_event_tool = create_calendar_event
check_availability_tool = check_calendar_availability
query_events_tool = query_calendar_events
update_event_tool = update_calendar_event
delete_event_tool = delete_calendar_event


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


# Song Generation Node
async def generate_song(state: EnhancedRAGState) -> EnhancedRAGState:
    """Generate song lyrics and audio if needed"""
    # Create an improved prompt for song generation
    song_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        Create a detailed and clear prompt for generating song lyrics based on 
        the user's request. Focus on theme, mood, style, and any specific requirements.
        Keep it concise but descriptive.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = song_prompt_template | llm | StrOutputParser()
    enhanced_prompt = await chain.ainvoke({"query": state["query"]})

    # Execute the song generation tool for lyrics
    song_lyrics = await song_tool.ainvoke(enhanced_prompt)
    state["song_lyrics"] = str(song_lyrics)

    # Generate audio using the Suno API
    try:
        suno_url = os.getenv("SUNO_URL")
        suno_api_key = os.getenv("SUNO_API_KEY")

        # Check if environment variables are set
        if not suno_url or not suno_api_key:
            print("Warning: SUNO_URL or SUNO_API_KEY environment variables not set")
            state["song_audio_url"] = ""
            state["song_audio_data"] = {"error": "Missing Suno API configuration"}
            return Command(goto="generate_response", update=state)

        payload = {
            "model": "minimax-music",
            "prompt": song_lyrics,
        }

        headers = {
            "Authorization": f"Bearer {suno_api_key}",
            "Content-Type": "application/json",
        }

        # Make request to Suno API
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(suno_url, json=payload, headers=headers)

                # Check if the request was successful
                if response.status_code == 200 or response.status_code == 201:
                    song_data = response.json()
                    print(f"Suno API response: {song_data}")

                    # Store the full response data for future reference
                    state["song_audio_data"] = song_data

                    # Extract the audio URL from the response (adjust based on actual Suno API response format)
                    if "audio_url" in song_data:
                        state["song_audio_url"] = song_data["audio_url"]
                    elif "url" in song_data:
                        state["song_audio_url"] = song_data["url"]
                    elif "id" in song_data:
                        # Some APIs return an ID first and then require a separate call to get the URL
                        # Try to handle both direct URL and ID-based reference
                        state["song_audio_url"] = f"{suno_url}/audio/{song_data['id']}"

                        # If the API returns a status indicating the song is being processed
                        # we could implement polling logic here, but for simplicity, we'll just
                        # pass the ID and let the frontend handle polling if needed
                        if "status" in song_data and song_data["status"] in [
                            "processing",
                            "pending",
                        ]:
                            print(
                                f"Song generation in progress with ID: {song_data['id']}"
                            )
                    else:
                        # Fallback: store a serialized version of the response
                        print(f"Unrecognized Suno API response format: {song_data}")
                        state["song_audio_url"] = f"suno:data:{json.dumps(song_data)}"
                else:
                    error_message = (
                        f"Suno API error: {response.status_code} - {response.text}"
                    )
                    print(error_message)
                    state["song_audio_url"] = ""
                    state["song_audio_data"] = {
                        "error": error_message,
                        "status_code": response.status_code,
                    }
            except httpx.TimeoutException:
                error_message = "Suno API request timed out"
                print(error_message)
                state["song_audio_url"] = ""
                state["song_audio_data"] = {"error": error_message, "status": "timeout"}
    except Exception as e:
        error_message = f"Error generating song audio: {str(e)}"
        print(error_message)
        state["song_audio_url"] = ""
        state["song_audio_data"] = {"error": error_message}

    return Command(
        goto="generate_response",
        update=state,
    )


# Video Generation Node
async def generate_video(state: EnhancedRAGState) -> EnhancedRAGState:
    """Generate a video if needed"""
    # Create an improved prompt for video generation
    video_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        Create a detailed and clear prompt for generating a video based on 
        the user's request. Focus on scene description, camera movement, style, and any specific visual elements.
        Include details about setting, characters or objects, actions, and atmosphere.
        Keep it concise (under 200 words) but descriptive.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = video_prompt_template | llm | StrOutputParser()
    enhanced_prompt = await chain.ainvoke({"query": state["query"]})

    # Execute the video generation tool
    video_result = await video_tool.ainvoke(enhanced_prompt)

    # Store the video data
    state["video_data"] = video_result

    # Extract URL if available
    if isinstance(video_result, dict) and "url" in video_result:
        state["video_url"] = video_result["url"]
    else:
        state["video_url"] = ""

    return Command(
        goto="generate_response",
        update=state,
    )


# Calendar Processing Node
async def process_calendar_request(state: EnhancedRAGState) -> EnhancedRAGState:
    """Process calendar-related requests including scheduling, availability, and queries"""

    # First, determine what type of calendar action is needed
    calendar_action_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are an AI assistant that analyzes calendar-related requests.
        Determine which calendar action is needed based on the user's query:
        
        1. CREATE: User wants to create a new event, schedule a meeting, or add something to their calendar
        2. QUERY: User wants to know about their schedule, upcoming events, or find specific events
        3. AVAILABILITY: User wants to check for free/busy times or find available slots
        4. UPDATE: User wants to modify an existing event
        5. DELETE: User wants to cancel or remove an event
        
        Output a single action word: CREATE, QUERY, AVAILABILITY, UPDATE, or DELETE.
        """,
            ),
            ("user", "{query}"),
        ]
    )

    chain = calendar_action_prompt | llm | StrOutputParser()
    calendar_action = await chain.ainvoke({"query": state["query"]})
    state["calendar_action"] = calendar_action.strip().upper()

    # Based on the action, extract relevant information and call appropriate tools
    if state["calendar_action"] == "CREATE":
        # Extract event details from query
        event_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Extract calendar event details from the user's query.
            Return a JSON object with these fields:
            - summary: The event title/name
            - description: Details about the event
            - start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - end_time: End time in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - attendees: Optional array of email addresses
            - location: Physical or virtual location (can be empty)
            
            If a field is not mentioned in the query, use null or empty string/array.
            For date/time, make reasonable assumptions if only partial information is given:
            - If no year is specified, use current year
            - If no date is specified, use next appropriate day
            - If no time is specified for a meeting, default to 30 minutes
            - Use UTC timezone for ISO format
            """,
                ),
                ("user", "{query}"),
            ]
        )

        # Extract event details
        event_extraction_chain = event_extraction_prompt | llm | JsonOutputParser()
        event_details = await event_extraction_chain.ainvoke({"query": state["query"]})

        # Create the event
        result = await create_event_tool.ainvoke({"event_details": event_details})
        state["calendar_result"] = result

    elif state["calendar_action"] == "QUERY":
        # Extract query parameters
        query_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Extract calendar query parameters from the user's request.
            Return a JSON object with these fields:
            - start_time: Start time for query range in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - end_time: End time for query range in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - max_results: Maximum number of events to return (default to 10)
            - search_term: Text to search for in events (or null)
            
            For date ranges, make reasonable assumptions:
            - If user asks about "today", use today's date
            - If user asks about "this week", use current week
            - If user asks about "next month", use appropriate range
            - Use UTC timezone for ISO format
            """,
                ),
                ("user", "{query}"),
            ]
        )

        # Extract query parameters
        query_extraction_chain = query_extraction_prompt | llm | JsonOutputParser()
        query_params = await query_extraction_chain.ainvoke({"query": state["query"]})

        # Query events
        result = await query_events_tool.ainvoke({"query_params": query_params})
        state["calendar_events"] = result.get("events", [])
        state["calendar_result"] = result

    elif state["calendar_action"] == "AVAILABILITY":
        # Extract time range for availability check
        availability_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Extract time range and attendees for an availability check from the user's query.
            Return a JSON object with these fields:
            - start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - end_time: End time in ISO format (YYYY-MM-DDTHH:MM:SS.000Z)
            - attendees: Array of email addresses (can be empty)
            
            Make reasonable assumptions about the time range:
            - If specific times are mentioned, use those
            - If only a date is mentioned, use business hours (9am-5pm)
            - If no date is specified, use next appropriate day
            - Use UTC timezone for ISO format
            """,
                ),
                ("user", "{query}"),
            ]
        )

        # Extract time range
        availability_extraction_chain = (
            availability_extraction_prompt | llm | JsonOutputParser()
        )
        time_range = await availability_extraction_chain.ainvoke(
            {"query": state["query"]}
        )

        # Check availability
        result = await check_availability_tool.ainvoke({"time_range": time_range})
        state["calendar_free_busy"] = result.get("alternatives", [])
        state["calendar_result"] = result

    elif state["calendar_action"] == "UPDATE":
        # Extract event update details
        update_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Extract event update details from the user's query.
            Return a JSON object with these fields:
            - event_id: ID of the event to update (can be null if not specified)
            - summary: New event title (or null if not changing)
            - description: New description (or null if not changing)
            - start_time: New start time in ISO format (or null if not changing)
            - end_time: New end time in ISO format (or null if not changing)
            - location: New location (or null if not changing)
            - attendees: Array of new attendee emails (or null if not changing)
            
            If the event_id is not specified, you will need to query for the event first.
            If the user refers to an event by name/date/time, indicate this in your response.
            """,
                ),
                ("user", "{query}"),
            ]
        )

        # Extract update details
        update_extraction_chain = update_extraction_prompt | llm | JsonOutputParser()
        event_update = await update_extraction_chain.ainvoke({"query": state["query"]})

        # Check if we need to find the event ID first
        if not event_update.get("event_id"):
            # Create query parameters to find the event
            search_term = event_update.get("summary", "")
            if not search_term:
                # Try to extract a search term from the query
                extract_term_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "Extract the name or key identifying words of the event the user wants to update.",
                        ),
                        ("user", "{query}"),
                    ]
                )
                term_chain = extract_term_prompt | llm | StrOutputParser()
                search_term = await term_chain.ainvoke({"query": state["query"]})

            # Query to find the event
            query_params = {
                "search_term": search_term,
                "max_results": 3,  # Limit to a few results
            }

            result = await query_events_tool.ainvoke({"query_params": query_params})
            events = result.get("events", [])

            if events:
                # Found events - ask for clarification if multiple
                if len(events) > 1:
                    state["calendar_events"] = events
                    state["calendar_result"] = {
                        "status": "needs_clarification",
                        "message": "Multiple matching events found, please specify which one to update",
                        "events": events,
                    }
                    return Command(goto="generate_response", update=state)
                else:
                    # Single event found, use it
                    event_update["event_id"] = events[0]["id"]
            else:
                # No events found
                state["calendar_result"] = {
                    "status": "error",
                    "message": "Could not find the event you want to update",
                }
                return Command(goto="generate_response", update=state)

        # Update the event
        result = await update_event_tool.ainvoke({"event_update": event_update})
        state["calendar_result"] = result

    elif state["calendar_action"] == "DELETE":
        # Extract event ID for deletion
        delete_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Extract event deletion details from the user's query.
            Return a JSON object with these fields:
            - event_id: ID of the event to delete (can be null if not specified)
            - notify_attendees: Boolean indicating whether to notify attendees
            
            If the event_id is not specified, you will need to query for the event first.
            If the user refers to an event by name/date/time, indicate this in your response.
            """,
                ),
                ("user", "{query}"),
            ]
        )

        # Extract deletion details
        delete_extraction_chain = delete_extraction_prompt | llm | JsonOutputParser()
        event_id_dict = await delete_extraction_chain.ainvoke({"query": state["query"]})

        # Check if we need to find the event ID first
        if not event_id_dict.get("event_id"):
            # Create query parameters to find the event
            extract_term_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Extract the name or key identifying words of the event the user wants to delete.",
                    ),
                    ("user", "{query}"),
                ]
            )
            term_chain = extract_term_prompt | llm | StrOutputParser()
            search_term = await term_chain.ainvoke({"query": state["query"]})

            # Query to find the event
            query_params = {
                "search_term": search_term,
                "max_results": 3,  # Limit to a few results
            }

            result = await query_events_tool.ainvoke({"query_params": query_params})
            events = result.get("events", [])

            if events:
                # Found events - ask for clarification if multiple
                if len(events) > 1:
                    state["calendar_events"] = events
                    state["calendar_result"] = {
                        "status": "needs_clarification",
                        "message": "Multiple matching events found, please specify which one to delete",
                        "events": events,
                    }
                    return Command(goto="generate_response", update=state)
                else:
                    # Single event found, use it
                    event_id_dict["event_id"] = events[0]["id"]
            else:
                # No events found
                state["calendar_result"] = {
                    "status": "error",
                    "message": "Could not find the event you want to delete",
                }
                return Command(goto="generate_response", update=state)

        # Delete the event
        result = await delete_event_tool.ainvoke({"event_id_dict": event_id_dict})
        state["calendar_result"] = result

    # After processing, move to response generation
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
            "If an image was generated, simply describe what's in the image and mention it's available for viewing. "
            "If a song was generated, mention that a song has been created for the user and is available for listening. "
            "If a video was generated, mention that a video has been created and is available for viewing."
            "If calendar information was retrieved or modified, present it in a clear, organized way.",
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

    if state["song_audio_url"] and state["song_audio_url"] != "":
        messages.append(
            (
                "system",
                "You have generated a song which is available at: "
                + state["song_audio_url"]
                + ". Mention that a song has been created for the user and DO include the URL in your response. "
                + "Tell them they can listen to it by clicking the link or using the audio player in the interface.",
            )
        )
    elif state["song_lyrics"] and state["song_lyrics"] != "":
        # We generated lyrics but no audio URL yet
        messages.append(
            (
                "system",
                "You have generated song lyrics but the audio is still being processed. "
                "Tell the user that a song is being generated and will be available shortly. "
                "Do not include the raw lyrics in your response.",
            )
        )

    if state["video_url"] and state["video_url"] != "":
        video_status = (
            state["video_data"].get("status", "ready")
            if isinstance(state["video_data"], dict)
            else "ready"
        )

        if video_status.lower() in ["processing", "pending"]:
            messages.append(
                (
                    "system",
                    "You have started generating a video based on the user's request. "
                    "Tell the user that their video is being processed and will be available shortly. "
                    "You can mention that video generation can take a minute or two to complete.",
                )
            )
        else:
            messages.append(
                (
                    "system",
                    "You have generated a video which is available at: "
                    + state["video_url"]
                    + ". Mention that a video has been created for the user and DO include the URL in your response. "
                    + "Tell them they can watch it by clicking the link or using the video player in the interface.",
                )
            )

    # Add calendar information if available
    if state.get("calendar_action") and state.get("calendar_result"):
        calendar_action = state.get("calendar_action", "")
        calendar_result = state.get("calendar_result", {})

        # Format calendar information based on action type
        if calendar_action == "CREATE" and calendar_result.get("status") == "success":
            messages.append(
                (
                    "system",
                    f"You have created a calendar event. The event has been added to the user's Google Calendar. "
                    f"The event link is: {calendar_result.get('event_link', 'Not available')}. "
                    f"Let the user know the event was created successfully and provide the link to view it.",
                )
            )
        elif calendar_action == "QUERY" and calendar_result.get("status") == "success":
            events = state.get("calendar_events", [])
            if events:
                events_text = "Here are the calendar events I found:\n\n"
                for i, event in enumerate(
                    events[:5]
                ):  # Limit to 5 events in the prompt
                    start_time = event.get("start", "")
                    # Try to format the date nicely if it's ISO format
                    try:
                        if "T" in start_time:
                            dt = datetime.fromisoformat(
                                start_time.replace("Z", "+00:00")
                            )
                            start_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
                    except:
                        pass  # Use the original format if parsing fails

                    events_text += f"{i+1}. {event.get('summary', 'Untitled Event')} - {start_time}\n"
                    if event.get("location"):
                        events_text += f"   Location: {event.get('location')}\n"

                if len(events) > 5:
                    events_text += f"\nPlus {len(events) - 5} more events..."

                messages.append(("system", events_text))
            else:
                messages.append(
                    (
                        "system",
                        "You queried the user's calendar but no events were found matching the criteria. "
                        "Let them know their schedule is clear for the specified time period.",
                    )
                )
        elif (
            calendar_action == "AVAILABILITY"
            and calendar_result.get("status") == "success"
        ):
            is_available = calendar_result.get("is_available", False)
            alternatives = state.get("calendar_free_busy", [])

            if is_available:
                messages.append(
                    (
                        "system",
                        "The requested time slot is available. Let the user know they are free during this time "
                        "and can schedule their meeting/event.",
                    )
                )
            else:
                if alternatives:
                    alt_text = "The requested time is not available, but here are some alternatives:\n\n"
                    for i, alt in enumerate(
                        alternatives[:3]
                    ):  # Limit to 3 alternatives
                        try:
                            start = datetime.fromisoformat(
                                alt.get("start", "").replace("Z", "+00:00")
                            )
                            end = datetime.fromisoformat(
                                alt.get("end", "").replace("Z", "+00:00")
                            )
                            alt_text += f"{i+1}. {start.strftime('%A, %B %d, %Y from %I:%M %p')} to {end.strftime('%I:%M %p')}\n"
                        except:
                            alt_text += (
                                f"{i+1}. {alt.get('start')} to {alt.get('end')}\n"
                            )

                    messages.append(("system", alt_text))
                else:
                    messages.append(
                        (
                            "system",
                            "The requested time is not available and no suitable alternatives were found. "
                            "Suggest the user try a different day or time range.",
                        )
                    )
        elif calendar_action == "UPDATE" and calendar_result.get("status") == "success":
            messages.append(
                (
                    "system",
                    f"You have updated the calendar event. The changes have been saved to the user's Google Calendar. "
                    f"The updated event link is: {calendar_result.get('event_link', 'Not available')}. "
                    f"Let the user know the event was updated successfully.",
                )
            )
        elif calendar_action == "DELETE" and calendar_result.get("status") == "success":
            messages.append(
                (
                    "system",
                    "You have deleted the calendar event. The event has been removed from the user's Google Calendar. "
                    "Let the user know the event was deleted successfully.",
                )
            )
        elif calendar_result.get("status") == "needs_clarification":
            # For cases where we need user clarification (multiple matching events)
            events = state.get("calendar_events", [])
            if events:
                events_text = "Multiple events match your description. Please ask the user to clarify which one they mean:\n\n"
                for i, event in enumerate(events):
                    try:
                        start_time = event.get("start", "")
                        if "T" in start_time:
                            dt = datetime.fromisoformat(
                                start_time.replace("Z", "+00:00")
                            )
                            start_time = dt.strftime("%A, %B %d at %I:%M %p")
                    except:
                        pass

                    events_text += f"{i+1}. {event.get('summary', 'Untitled Event')} - {start_time}\n"

                messages.append(("system", events_text))
        elif calendar_result.get("status") == "error":
            error_message = calendar_result.get("message", "Unknown error")
            # Escape any curly braces in the error message
            error_message = error_message.replace("{", "{{").replace("}", "}}")

            messages.append(
                (
                    "system",
                    f"There was an error with the calendar operation: {error_message}. "
                    "Apologize to the user and suggest they try again with more specific details.",
                )
            )

    # Add only the current query - don't include message history to avoid confusion
    messages.append(("user", state["query"]))

    # Add this before the prompt line
    for i, msg in enumerate(messages):
        print(f"Message {i}: {msg}")

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({})

    state["response"] = response
    print(f"Debug: Response generated: {response}")
    return state


# Build the graph
def build_enhanced_graph():
    workflow = StateGraph(EnhancedRAGState)

    # Add nodes
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("retrieve_from_rag", retrieve_from_rag)
    workflow.add_node("perform_web_search", perform_web_search)
    workflow.add_node("generate_image", generate_image)
    workflow.add_node("generate_song", generate_song)
    workflow.add_node("generate_video", generate_video)
    workflow.add_node("process_calendar_request", process_calendar_request)
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
                "use_song_gen": False,
                "use_video_gen": False,
                "use_calendar": False,
            },
            "rag_context": "",
            "search_results": "",
            "image_url": "",
            "song_lyrics": "",
            "song_audio_url": "",
            "song_audio_data": {},
            "video_url": "",
            "video_data": {},
            "response": "",
            # Calendar-related fields
            "calendar_action": "",
            "calendar_events": [],
            "calendar_free_busy": [],
            "calendar_result": {},
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
                "use_song_gen": False,
                "use_video_gen": False,
                "use_calendar": False,
            },
            "rag_context": "",
            "search_results": "",
            "image_url": "",
            "song_lyrics": "",
            "song_audio_url": "",
            "song_audio_data": {},
            "video_url": "",
            "video_data": {},
            "response": "",
            # Calendar-related fields
            "calendar_action": "",
            "calendar_events": [],
            "calendar_free_busy": [],
            "calendar_result": {},
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

                if tool_decision.get("use_song_gen", False):
                    safe_query = (
                        query.replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type":"song_gen_start","query":"{safe_query}"}}\n\n'

                if tool_decision.get("use_video_gen", False):
                    safe_query = (
                        query.replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type":"video_gen_start","query":"{safe_query}"}}\n\n'

                if tool_decision.get("use_calendar", False):
                    safe_query = (
                        query.replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type":"calendar_start","query":"{safe_query}"}}\n\n'

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

        # Song generation results
        elif event_type == "on_node_end" and event.get("name") == "generate_song":
            song_audio_url = event["data"]["output"].get("song_audio_url", "")
            song_audio_data = event["data"]["output"].get("song_audio_data", {})

            if song_audio_url and song_audio_url != "":
                # Create a response object with just the audio URL
                song_response = {
                    "type": "song_generated",
                    "audio_url": song_audio_url,
                }

                # Include essential audio data if available
                if song_audio_data:
                    # Include only necessary fields to avoid overwhelming the response
                    if isinstance(song_audio_data, dict):
                        filtered_data = {}
                        for key in ["id", "status", "progress", "created_at", "url"]:
                            if key in song_audio_data:
                                filtered_data[key] = song_audio_data[key]

                        # Some APIs might nest data, try to extract it
                        if "data" in song_audio_data and isinstance(
                            song_audio_data["data"], dict
                        ):
                            for key in ["id", "status", "progress", "url", "audio_url"]:
                                if key in song_audio_data["data"]:
                                    filtered_data[key] = song_audio_data["data"][key]

                        song_response["audio_data"] = filtered_data

                # Convert to JSON string and send to client
                song_response_json = json.dumps(song_response)
                yield f"data: {song_response_json}\n\n"
                print("Debug: Song audio URL sent to frontend")

        # Video generation results
        elif event_type == "on_node_end" and event.get("name") == "generate_video":
            video_url = event["data"]["output"].get("video_url", "")
            video_data = event["data"]["output"].get("video_data", {})

            # Prepare response with video information
            video_response = {
                "type": "video_generated",
            }

            if video_url and video_url != "":
                video_response["video_url"] = video_url

            if video_data:
                # Include relevant video metadata
                if isinstance(video_data, dict):
                    filtered_data = {}
                    # Extract important fields
                    for key in ["id", "status", "progress", "estimated_time"]:
                        if key in video_data:
                            filtered_data[key] = video_data[key]

                    video_response["video_data"] = filtered_data

                    # If video is still processing, include that information
                    if video_data.get("status", "").lower() in [
                        "processing",
                        "pending",
                    ]:
                        video_response["status"] = "processing"
                        if "estimated_time" in video_data:
                            video_response["estimated_time"] = video_data[
                                "estimated_time"
                            ]

            # Send video information to the client
            video_response_json = json.dumps(video_response)
            yield f"data: {video_response_json}\n\n"
            print(f"Debug: Video generation info sent to frontend: {video_response}")

        # Calendar processing results
        elif (
            event_type == "on_node_end"
            and event.get("name") == "process_calendar_request"
        ):
            calendar_action = event["data"]["output"].get("calendar_action", "")
            calendar_result = event["data"]["output"].get("calendar_result", {})

            if calendar_result:
                # Prepare calendar response object
                calendar_response = {
                    "type": "calendar_result",
                    "action": calendar_action,
                    "status": calendar_result.get("status", "unknown"),
                }

                # Add action-specific data
                if (
                    calendar_action == "CREATE"
                    and calendar_result.get("status") == "success"
                ):
                    calendar_response["event_id"] = calendar_result.get("event_id", "")
                    calendar_response["event_link"] = calendar_result.get(
                        "event_link", ""
                    )

                elif (
                    calendar_action == "QUERY"
                    and calendar_result.get("status") == "success"
                ):
                    calendar_events = event["data"]["output"].get("calendar_events", [])
                    if calendar_events and len(calendar_events) > 0:
                        # Limit events data to essential fields to keep response size manageable
                        simplified_events = []
                        for evt in calendar_events:
                            simplified_events.append(
                                {
                                    "id": evt.get("id", ""),
                                    "summary": evt.get("summary", ""),
                                    "start": evt.get("start", ""),
                                    "end": evt.get("end", ""),
                                    "location": evt.get("location", ""),
                                }
                            )
                        calendar_response["events"] = simplified_events
                        calendar_response["event_count"] = len(simplified_events)

                elif (
                    calendar_action == "AVAILABILITY"
                    and calendar_result.get("status") == "success"
                ):
                    calendar_response["is_available"] = calendar_result.get(
                        "is_available", False
                    )
                    calendar_free_busy = event["data"]["output"].get(
                        "calendar_free_busy", []
                    )
                    if calendar_free_busy:
                        calendar_response["alternatives"] = calendar_free_busy

                elif (
                    calendar_action in ["UPDATE", "DELETE"]
                    and calendar_result.get("status") == "success"
                ):
                    if "event_id" in calendar_result:
                        calendar_response["event_id"] = calendar_result["event_id"]
                    if "event_link" in calendar_result:
                        calendar_response["event_link"] = calendar_result["event_link"]

                # Handle cases needing clarification (multiple matching events)
                elif calendar_result.get("status") == "needs_clarification":
                    calendar_events = event["data"]["output"].get("calendar_events", [])
                    if calendar_events:
                        simplified_events = []
                        for evt in calendar_events:
                            simplified_events.append(
                                {
                                    "id": evt.get("id", ""),
                                    "summary": evt.get("summary", ""),
                                    "start": evt.get("start", ""),
                                    "end": evt.get("end", ""),
                                    "location": evt.get("location", ""),
                                }
                            )
                        calendar_response["events"] = simplified_events
                        calendar_response["needs_clarification"] = True

                # Send calendar information to client
                try:
                    calendar_response_json = json.dumps(calendar_response)
                    yield f"data: {calendar_response_json}\n\n"
                    print(f"Debug: Calendar result sent to frontend: {calendar_action}")
                except Exception as e:
                    print(f"Error sending calendar data: {str(e)}")

        # Response completion notification
        elif event_type == "on_node_end" and event.get("name") == "generate_response":
            output_data = event["data"]["output"]
            has_image = (
                bool(output_data.get("image_url"))
                and output_data.get("image_url") != ""
            )
            has_song_audio = (
                bool(output_data.get("song_audio_url"))
                and output_data.get("song_audio_url") != ""
            )
            has_video = (
                bool(output_data.get("video_url"))
                and output_data.get("video_url") != ""
            )
            has_calendar = bool(output_data.get("calendar_result"))

            response_data = {
                "type": "response_complete",
                "has_image": False,
                "has_song_audio": False,
                "has_video": False,
                "has_calendar": False,
            }

            if has_image:
                response_data["has_image"] = True
                response_data["image_url"] = output_data["image_url"]

            if has_song_audio:
                response_data["has_song_audio"] = True
                response_data["song_audio_url"] = output_data["song_audio_url"]

                # Include selected audio data fields if available
                if "song_audio_data" in output_data and output_data["song_audio_data"]:
                    audio_data = output_data["song_audio_data"]
                    if isinstance(audio_data, dict):
                        # Only include essential fields to keep the response size manageable
                        essential_fields = {}
                        for key in ["id", "status", "progress", "url"]:
                            if key in audio_data:
                                essential_fields[key] = audio_data[key]

                        # Check if audio data is nested within a data field
                        if "data" in audio_data and isinstance(
                            audio_data["data"], dict
                        ):
                            for key in ["id", "status", "progress", "url", "audio_url"]:
                                if key in audio_data["data"]:
                                    essential_fields[key] = audio_data["data"][key]

                        response_data["song_audio_data"] = essential_fields

            if has_video:
                response_data["has_video"] = True
                response_data["video_url"] = output_data["video_url"]

                # Include video processing status info
                if "video_data" in output_data and output_data["video_data"]:
                    video_data = output_data["video_data"]
                    if isinstance(video_data, dict):
                        # Extract key video information
                        video_info = {}
                        for key in ["id", "status", "progress", "estimated_time"]:
                            if key in video_data:
                                video_info[key] = video_data[key]

                        response_data["video_data"] = video_info

            if has_calendar:
                response_data["has_calendar"] = True
                response_data["calendar_action"] = output_data.get(
                    "calendar_action", ""
                )

                # Include minimal calendar result info
                calendar_result = output_data.get("calendar_result", {})
                if calendar_result:
                    response_data["calendar_status"] = calendar_result.get("status", "")

                    # Include event link if available
                    if "event_link" in calendar_result:
                        response_data["calendar_event_link"] = calendar_result[
                            "event_link"
                        ]

            response_json = json.dumps(response_data)
            yield f"data: {response_json}\n\n"

    # Store conversation
    calendar_event = None

    # If there was a calendar creation or update, store event details
    if (
        state.get("calendar_action") in ["CREATE", "UPDATE"]
        and state.get("calendar_result", {}).get("status") == "success"
    ):
        # For calendar events that were created or updated, extract event details
        calendar_result = state.get("calendar_result", {})
        event_id = calendar_result.get("event_id", "")

        if event_id:
            # Query the event details if we have successful creation/update
            try:
                service = get_calendar_service()
                if service:
                    event = (
                        service.events()
                        .get(calendarId="primary", eventId=event_id)
                        .execute()
                    )
                    if event:
                        # Extract key event details
                        start = event.get("start", {})
                        end = event.get("end", {})

                        calendar_event = {
                            "id": event_id,
                            "summary": event.get("summary", "Untitled Event"),
                            "start": start.get("dateTime", start.get("date", "")),
                            "end": end.get("dateTime", end.get("date", "")),
                            "location": event.get("location", ""),
                            "attendees": [
                                attendee.get("email", "")
                                for attendee in event.get("attendees", [])
                            ],
                        }
            except Exception as e:
                print(f"Error retrieving event details for storage: {str(e)}")

    await store_conversation(
        conversation_id=state["conversation_id"],
        human_message=query,
        ai_response=state.get("response", "No response generated"),
        calendar_event=calendar_event,
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


# Add new endpoints for chat history management


@app.get("/conversations")
async def get_conversations():
    """Get list of unique conversation IDs with their dates and topics"""
    try:
        # Query for distinct conversation IDs with their dates
        query_response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(),
            limit=1000,  # Adjust limit as needed
        )

        # Process results to extract unique conversation sessions
        conversations = {}
        for point in query_response[0]:
            if point.payload and "conversation_id" in point.payload:
                conv_id = point.payload["conversation_id"]
                timestamp = point.payload.get("timestamp", "")
                date = point.payload.get("date", "")
                topic = point.payload.get("topic", "")

                if not date and timestamp:
                    # Extract date from timestamp if date field is missing
                    try:
                        date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
                    except:
                        date = "Unknown"

                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "id": conv_id,
                        "date": date,
                        "last_updated": timestamp,
                        "topic": topic,
                    }
                elif timestamp > conversations[conv_id]["last_updated"]:
                    conversations[conv_id]["last_updated"] = timestamp
                    # Only update topic if it's not already set and we have a new one
                    if not conversations[conv_id]["topic"] and topic:
                        conversations[conv_id]["topic"] = topic

        # Group conversations by date
        conversations_by_date = {}
        for conv in conversations.values():
            date = conv["date"]
            if date not in conversations_by_date:
                conversations_by_date[date] = []
            conversations_by_date[date].append(conv)

        # Sort each date's conversations by last_updated
        for date in conversations_by_date:
            conversations_by_date[date].sort(
                key=lambda x: x["last_updated"], reverse=True
            )

        # Sort dates in reverse chronological order
        sorted_dates = sorted(conversations_by_date.keys(), reverse=True)

        result = []
        for date in sorted_dates:
            result.append({"date": date, "sessions": conversations_by_date[date]})

        return result

    except Exception as e:
        return {"error": f"Failed to fetch conversations: {str(e)}"}


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get complete history for a specific conversation"""
    try:
        # Query for messages in this conversation
        query_response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(value=conversation_id),
                    ),
                ]
            ),
            limit=1000,  # Adjust limit as needed
        )

        # Extract and format messages
        messages = []
        for point in query_response[0]:
            if (
                point.payload
                and "type" in point.payload
                and point.payload["type"] == "conversation"
            ):
                content = point.payload.get("page_content", "")
                timestamp = point.payload.get("timestamp", "")

                # Extract user message and AI response if possible
                user_msg = ""
                ai_msg = ""
                if content:
                    parts = content.split("\nAssistant: ")
                    if len(parts) == 2:
                        user_part = parts[0].split("User: ")
                        if len(user_part) == 2:
                            user_msg = user_part[1].strip()
                        ai_msg = parts[1].strip()

                messages.append(
                    {
                        "timestamp": timestamp,
                        "user_message": user_msg,
                        "ai_response": ai_msg,
                        "full_content": content,
                    }
                )

        # Sort messages by timestamp
        messages.sort(key=lambda x: x["timestamp"])

        return {"conversation_id": conversation_id, "messages": messages}

    except Exception as e:
        return {"error": f"Failed to fetch conversation history: {str(e)}"}


@app.get("/search_conversations")
async def search_conversations(query: str, limit: int = 5):
    """Search across all conversations"""
    try:
        # Use vector search to find relevant messages
        docs = await vectorstore.asimilarity_search(query, k=limit)

        # Format results
        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "conversation_id": doc.metadata.get("conversation_id", ""),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "date": doc.metadata.get("date", ""),
                }
            )

        return {"query": query, "results": results}

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation by ID"""
    try:
        # Delete all points with this conversation_id
        delete_result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="conversation_id",
                            match=models.MatchValue(value=conversation_id),
                        ),
                    ]
                )
            ),
        )

        return {"success": True, "message": f"Conversation {conversation_id} deleted"}

    except Exception as e:
        return {"error": f"Failed to delete conversation: {str(e)}"}


# For direct execution of the script
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
