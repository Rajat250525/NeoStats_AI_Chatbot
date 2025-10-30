from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

class GroqChatModel:
    """
    Wrapper class for Groq LLM integration using LangChain.
    Handles real-time chat generation with context and system prompts.
    """
    def __init__(self, api_key, model_name="llama-3.1-8b-instant"):
        """
        Initialize the Groq chat model.

        Args:
            api_key (str): Your Groq API key.
            model_name (str): Groq model name (e.g., 'llama-3.1-8b-instant').
        """
        self.model_name = model_name
        self.client = ChatGroq(model=model_name, groq_api_key=api_key)

    def generate(self, user_input, context=None, system_prompt="You are a helpful AI assistant."):
        """
        Generate a response from the Groq LLM.

        Args:
            user_input (str): The latest user query.
            context (str, optional): Extra context (e.g., from PDF RAG).
            system_prompt (str, optional): System-level behavior instructions.

        Returns:
            str: The AI-generated response text.
        """
        try:
            # Prepare messages for the chat model
            messages = [SystemMessage(content=system_prompt)]
            if context:
                messages.append(HumanMessage(content=f"Context: {context}"))
            messages.append(HumanMessage(content=user_input))

            # ✅ Correct method to call Groq LLM
            response = self.client.invoke(messages)

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            return f"Error calling Groq model: {e}"


def get_chatgroq_model(api_key, model_name="llama-3.3-70b-versatile"):

    """
    Factory method to initialize the GroqChatModel.
    Returns None if API key is missing.
    """
    if not api_key:
        print("⚠️ No Groq API key provided. Model not initialized.")
        return None
    return GroqChatModel(api_key, model_name)
