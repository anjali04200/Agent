from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio
from typing import Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Environment variables (Consider moving to .env file for security)
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCWDJLP1hox4sEwhXmIZgHrPPvX7kJm6eA"

# Global configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

class AgentManager:
    """Manages the LangChain agent with configurable parameters"""
    
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        
        # Initialize FAISS vector store with placeholder
        self.vector_store = FAISS.from_documents(
            [Document(page_content="initial", metadata={"source": "system"})], 
            self.embedding_model
        )
        
        # Initialize with default LLM
        self.llm = None
        self.agent = None
        self.memory = None
        self._initialize_agent()
    
    def _initialize_agent(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
        """Initialize or reinitialize the agent with new parameters"""
        try:
            # Create LLM with specified parameters
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Setup tools
            tools = [
                Tool(
                    name="WebSearch", 
                    func=self._web_search_wrapper, 
                    description=(
                        "Use this tool for retrieving up-to-date information from the internet, "
                        "including current events, news, stock prices, sports scores, and weather forecasts."
                    )
                ),
                Tool(
                    name="Summarizer", 
                    func=self._summarizer_wrapper, 
                    description="Summarizes text content efficiently."
                ),
                Tool(
                    name="MemorySaver", 
                    func=self.save_to_memory, 
                    description="Stores important information into long-term memory."
                ),
                Tool(
                    name="MemoryQA", 
                    func=self.memory_qa, 
                    description="Queries stored memory for relevant information."
                ),
            ]
            
            # Setup memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                max_token_limit=2000  # Prevent memory overflow
            )
            
            # Initialize agent
            self.agent = initialize_agent(
                tools, 
                self.llm, 
                agent="zero-shot-react-description", 
                memory=self.memory, 
                verbose=True,
                max_iterations=5,  # Prevent infinite loops
                early_stopping_method="generate"
            )
            
            logger.info(f"Agent initialized with temperature={temperature}, max_tokens={max_tokens}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def _web_search_wrapper(self, query: str) -> str:
        """Wrapper for web search with error handling"""
        try:
            search_tool = TavilySearchResults()
            results = search_tool.run(query)
            return str(results)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"
    
    def _summarizer_wrapper(self, text: str) -> str:
        """Wrapper for text summarization with error handling"""
        try:
            if not text.strip():
                return "No content to summarize."
            
            summary_prompt = PromptTemplate(
                input_variables=["text"],
                template="Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
            )
            
            response = self.llm.invoke(summary_prompt.format(text=text))
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Summarization failed: {str(e)}"
    
    def save_to_memory(self, content: str) -> str:
        """Save content to vector store memory"""
        try:
            if not content.strip():
                return "No content provided to save."
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "memory",
                    "timestamp": str(asyncio.get_event_loop().time())
                }
            )
            self.vector_store.add_documents([doc])
            logger.info(f"Saved to memory: {content[:50]}...")
            return "Successfully saved to memory."
        except Exception as e:
            logger.error(f"Memory save error: {e}")
            return f"Failed to save to memory: {str(e)}"
    
    def memory_qa(self, query: str) -> str:
        """Query the vector store memory"""
        try:
            if not query.strip():
                return "No query provided."
            
            docs = self.vector_store.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in memory."
            
            chain = load_qa_with_sources_chain(self.llm, chain_type="stuff")
            result = chain.run(input_documents=docs, question=query)
            return result
        except Exception as e:
            logger.error(f"Memory QA error: {e}")
            return f"Memory query failed: {str(e)}"
    
    def update_configuration(self, temperature: float = None, max_tokens: int = None):
        """Update agent configuration dynamically"""
        current_temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
        current_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        
        # Reinitialize agent with new parameters
        self._initialize_agent(current_temp, current_tokens)
    
    def run_query(self, query: str, conversation_history: Optional[List[Tuple]] = None) -> str:
        """Run a query through the agent"""
        try:
            if not query.strip():
                return "Please provide a valid question or request."
            
            # Add conversation context if provided
            if conversation_history:
                context = self._format_conversation_context(conversation_history)
                if context:
                    query = f"Context from recent conversation:\n{context}\n\nCurrent question: {query}"
            
            logger.info(f"Processing query: {query[:100]}...")
            result = self.agent.run(query)
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _format_conversation_context(self, history: List[Tuple], max_context: int = 5) -> str:
        """Format recent conversation history for context"""
        try:
            if not history:
                return ""
            
            # Get last few exchanges for context
            recent_history = history[-max_context*2:] if len(history) > max_context*2 else history
            context_parts = []
            
            for item in recent_history:
                if len(item) >= 2:
                    role, message = item[0], item[1]
                    if role in ["user", "assistant"] and message.strip():
                        context_parts.append(f"{role.capitalize()}: {message}")
            
            return "\n".join(context_parts) if context_parts else ""
        except Exception as e:
            logger.error(f"Context formatting error: {e}")
            return ""
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.error(f"Memory clear error: {e}")
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        try:
            return {
                "conversation_length": len(self.memory.buffer_as_messages) if self.memory else 0,
                "vector_store_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            }
        except Exception as e:
            logger.error(f"Memory stats error: {e}")
            return {"error": str(e)}

# Global agent manager instance
agent_manager = AgentManager()

def ask_agent(
    query: str, 
    temperature: float = None, 
    max_tokens: int = None,
    conversation_history: Optional[List[Tuple]] = None
) -> str:
    """
    Main function to ask the agent a question with optional configuration
    
    Args:
        query: The question or request
        temperature: LLM temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        conversation_history: Recent conversation context
    
    Returns:
        Agent response string
    """
    try:
        # Update configuration if parameters provided
        if temperature is not None or max_tokens is not None:
            agent_manager.update_configuration(temperature, max_tokens)
        
        # Run the query
        return agent_manager.run_query(query, conversation_history)
        
    except Exception as e:
        logger.error(f"ask_agent error: {e}")
        return f"I apologize, but I'm currently experiencing technical difficulties: {str(e)}"

# Utility functions for Streamlit integration
def clear_agent_memory():
    """Clear the agent's conversation memory"""
    agent_manager.clear_memory()

def get_agent_stats() -> dict:
    """Get agent memory statistics"""
    return agent_manager.get_memory_stats()

def reset_agent():
    """Reset agent to default configuration"""
    global agent_manager
    agent_manager = AgentManager()

# For testing
if __name__ == "__main__":
    # Test the agent
    test_query = "Who is the Vice President of India?"
    print("Testing agent...")
    response = ask_agent(test_query)
    print(f"\n🤖 Response: {response}")
