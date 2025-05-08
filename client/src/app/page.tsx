"use client";

import Header from "@/components/Header";
import InputBar from "@/components/InputBar";
import MessageArea from "@/components/MessageArea";
import React, { useState } from "react";

interface SearchInfo {
  stages: string[];
  query: string;
  urls: string[];
}

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  searchInfo?: SearchInfo;
  imageUrl?: string;
}

const Home = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: "Hi there, how can I help you?",
      isUser: false,
      type: "message",
    },
  ]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (currentMessage.trim()) {
      // First add the user message to the chat
      const newMessageId =
        messages.length > 0
          ? Math.max(...messages.map((msg) => msg.id)) + 1
          : 1;

      setMessages((prev) => [
        ...prev,
        {
          id: newMessageId,
          content: currentMessage,
          isUser: true,
          type: "message",
        },
      ]);

      const userInput = currentMessage;
      setCurrentMessage(""); // Clear input field immediately

      try {
        // Create AI response placeholder
        const aiResponseId = newMessageId + 1;
        setMessages((prev) => [
          ...prev,
          {
            id: aiResponseId,
            content: "",
            isUser: false,
            type: "message",
            isLoading: true,
            searchInfo: {
              stages: [],
              query: "",
              urls: [],
            },
          },
        ]);

        // Create URL with conversation ID if it exists
        let url;
        if (conversationId) {
          // Use the continue endpoint when we have a conversation ID
          // url = `https://perplexity-2-0.onrender.com/rag_chat/continue/${encodeURIComponent(
          url = `http://localhost:8000/rag_chat/continue/${encodeURIComponent(
            conversationId
          )}/${encodeURIComponent(userInput)}`;
        } else {
          // Start a new conversation
          // url = `https://perplexity-2-0.onrender.com/rag_chat/${encodeURIComponent(
          url = `http://localhost:8000/rag_chat/${encodeURIComponent(
            userInput
          )}`;
        }

        // Connect to SSE endpoint using EventSource
        const eventSource = new EventSource(url);
        let streamedContent = "";
        let searchData: SearchInfo | null = null;
        let hasReceivedContent = false;
        let imageUrl: string | undefined = undefined;

        // Process incoming messages
        eventSource.onmessage = (event) => {
          try {
            const data: any = JSON.parse(event.data);

            if (data.type === "conversation_start") {
              // Store the conversation ID for future requests
              setConversationId(data.conversation_id);
            } else if (data.type === "content") {
              // Filter out all metadata and tool decision information
              let cleanContent = data.content;

              // Remove tool decision JSON-like patterns
              cleanContent = cleanContent.replace(
                /\{[\s]*use_(?:rag|search|image_gen)[\s\w,.:'"{}]*\}/g,
                ""
              );

              // Remove any other JSON-like metadata at the beginning
              cleanContent = cleanContent.replace(/^\s*\{[^}]*\}\s*/g, "");

              // Clean up any "waiting for response" text if we're getting actual content
              if (
                cleanContent.trim() &&
                streamedContent.includes("Waiting for response")
              ) {
                streamedContent = "";
              }

              streamedContent += cleanContent;
              hasReceivedContent = true;

              // Update message with accumulated content
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? { ...msg, content: streamedContent, isLoading: false }
                    : msg
                )
              );
            } else if (data.type === "search_start") {
              // Create search info with 'searching' stage
              const newSearchInfo = {
                stages: ["searching"],
                query: data.query,
                urls: [],
              };
              searchData = newSearchInfo;

              // Update the AI message with search info
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: streamedContent,
                        searchInfo: newSearchInfo,
                        isLoading: false,
                      }
                    : msg
                )
              );
            } else if (data.type === "search_results") {
              try {
                // Parse URLs from search results
                const urls =
                  typeof data.urls === "string"
                    ? JSON.parse(data.urls)
                    : data.urls;

                // Update search info to add 'reading' stage (don't replace 'searching')
                const newSearchInfo = {
                  stages: searchData
                    ? [...searchData.stages, "reading"]
                    : ["reading"],
                  query: searchData?.query || "",
                  urls: urls,
                };
                searchData = newSearchInfo;

                // Update the AI message with search info
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          content: streamedContent,
                          searchInfo: newSearchInfo,
                          isLoading: false,
                        }
                      : msg
                  )
                );
              } catch (err) {
                console.error("Error parsing search results:", err);
              }
            } else if (data.type === "image_gen_start") {
              // Handle image generation start
              const newSearchInfo = {
                stages: searchData
                  ? [...searchData.stages, "generating image"]
                  : ["generating image"],
                query: data.query,
                urls: searchData?.urls || [],
              };
              searchData = newSearchInfo;

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: streamedContent,
                        searchInfo: newSearchInfo,
                        isLoading: false,
                      }
                    : msg
                )
              );
            } else if (data.type === "image_generated") {
              // Handle image generation completion
              imageUrl = data.url;
              const newSearchInfo = {
                stages: searchData
                  ? [...searchData.stages, "image created"]
                  : ["image created"],
                query: searchData?.query || "",
                urls: searchData?.urls || [],
              };
              searchData = newSearchInfo;

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: streamedContent,
                        searchInfo: newSearchInfo,
                        isLoading: false,
                        imageUrl: imageUrl,
                      }
                    : msg
                )
              );
            } else if (data.type === "search_error") {
              // Handle search error
              const newSearchInfo = {
                stages: searchData
                  ? [...searchData.stages, "error"]
                  : ["error"],
                query: searchData?.query || "",
                error: data.error,
                urls: [],
              };
              searchData = newSearchInfo;

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: streamedContent,
                        searchInfo: newSearchInfo,
                        isLoading: false,
                      }
                    : msg
                )
              );
            } else if (data.type === "response_complete") {
              // Handle response completion with possible image
              if (data.has_image && data.image_url) {
                imageUrl = data.image_url;

                // Update message with image URL
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          content: streamedContent,
                          imageUrl: imageUrl,
                          isLoading: false,
                        }
                      : msg
                  )
                );
              }
            } else if (data.type === "end") {
              // When stream ends, add 'writing' stage if we had search info
              if (searchData) {
                const finalSearchInfo = {
                  ...searchData,
                  stages: [...searchData.stages, "writing"],
                };

                // Final clean-up of content to remove any leftover metadata
                const finalCleanContent = streamedContent
                  .replace(/^\s*\{[^}]*\}\s*/g, "") // Remove JSON-like metadata at the beginning
                  .trim();

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          content: finalCleanContent,
                          searchInfo: finalSearchInfo,
                          isLoading: false,
                          imageUrl: imageUrl,
                        }
                      : msg
                  )
                );
              } else {
                // If no search data, just clean the content
                const finalCleanContent = streamedContent
                  .replace(/^\s*\{[^}]*\}\s*/g, "") // Remove JSON-like metadata at the beginning
                  .trim();

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          content: finalCleanContent,
                          isLoading: false,
                          imageUrl: imageUrl,
                        }
                      : msg
                  )
                );
              }

              eventSource.close();
            } else if (data.type === "error") {
              // Handle explicit error messages from the server
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: `Error: ${data.message || "Unknown error"}`,
                        isLoading: false,
                      }
                    : msg
                )
              );
              eventSource.close();
            }
          } catch (error) {
            console.error("Error parsing event data:", error, event.data);
          }
        };

        // Handle errors
        eventSource.onerror = (error) => {
          console.error("EventSource error:", error);
          eventSource.close();

          // Only update with error if we don't have content yet
          if (!streamedContent) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiResponseId
                  ? {
                      ...msg,
                      content:
                        "Sorry, there was an error processing your request.",
                      isLoading: false,
                    }
                  : msg
              )
            );
          }
        };

        // Listen for end event
        eventSource.addEventListener("end", () => {
          eventSource.close();
        });
      } catch (error) {
        console.error("Error setting up EventSource:", error);
        setMessages((prev) => [
          ...prev,
          {
            id: newMessageId + 1,
            content: "Sorry, there was an error connecting to the server.",
            isUser: false,
            type: "message",
            isLoading: false,
          },
        ]);
      }
    }
  };

  // Add a function to reset the conversation
  const resetConversation = () => {
    setConversationId(null);
    setMessages([
      {
        id: 1,
        content: "Hi there, how can I help you?",
        isUser: false,
        type: "message",
      },
    ]);
  };

  return (
    <div className="flex justify-center bg-gray-100 min-h-screen py-8 px-4">
      {/* Main container with refined shadow and border */}
      <div className="w-[70%] bg-white flex flex-col rounded-xl shadow-lg border border-gray-100 overflow-hidden h-[90vh]">
        <Header onNewChat={resetConversation} />
        <MessageArea messages={messages} />
        <InputBar
          currentMessage={currentMessage}
          setCurrentMessage={setCurrentMessage}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  );
};

export default Home;
