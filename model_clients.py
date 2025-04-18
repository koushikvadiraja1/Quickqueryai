"""
Model Clients for different AI providers.
This module provides a unified interface for interacting with different AI model providers.
"""

import os
import json
import requests
import logging
import streamlit as st
import ollama

# Import utility functions for API key management
from utils import load_env_vars, get_api_key, save_api_key

# Load environment variables from .env file
load_env_vars()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelClient:
    """Base class for model clients"""
    def __init__(self, model_name):
        self.model_name = model_name
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """
        Generate a completion from the model.
        
        Args:
            messages: List of message dictionaries (role, content)
            temperature: Creativity of response (0.0 to 1.0)
            response_format: Format for the response (e.g., {"type": "json_object"})
            
        Returns:
            Generated text or JSON response
        """
        raise NotImplementedError("Subclasses must implement generate_completion")
    
    def generate_streaming_completion(self, messages, temperature=0.7, response_format=None):
        """
        Generate a streaming completion from the model.
        
        Args:
            messages: List of message dictionaries (role, content)
            temperature: Creativity of response (0.0 to 1.0)
            response_format: Format for the response (e.g., {"type": "json_object"})
            
        Returns:
            Generator yielding chunks of the response as they become available
        """
        raise NotImplementedError("Subclasses must implement generate_streaming_completion")


class OpenAIClient(ModelClient):
    """Client for OpenAI models"""
    def __init__(self, model_name="gpt-4o"):
        super().__init__(model_name)
        
        # Import OpenAI library
        try:
            from openai import OpenAI
            
            # Get API key using our utility function
            self.api_key = get_api_key("openai")
                
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Please provide it in the sidebar settings.")
            
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'")
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """Generate completion using OpenAI API"""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
            
    def generate_streaming_completion(self, messages, temperature=0.7, response_format=None):
        """Generate streaming completion using OpenAI API"""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True  # Enable streaming
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            # Create a streaming response
            stream = self.client.chat.completions.create(**kwargs)
            
            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"OpenAI API streaming error: {str(e)}"


class AnthropicClient(ModelClient):
    """Client for Anthropic Claude models"""
    def __init__(self, model_name="claude-3-5-sonnet-20241022"):
        # Note: The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        # Always prefer using claude-3-5-sonnet-20241022 as it is the latest model
        super().__init__(model_name)
        
        # Get API key using our utility function
        self.api_key = get_api_key("anthropic")
            
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please provide it in the sidebar settings.")
        
        # Default API endpoint
        self.api_endpoint = "https://api.anthropic.com/v1/messages"
    
    def _format_messages(self, messages):
        """Convert OpenAI-style messages to Anthropic format"""
        # Extract system message if present
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        
        # Construct Anthropic messages format
        formatted_messages = []
        
        # If there's a system message, add it as a user message at the beginning
        if system_message:
            user_content = f"System instruction: {system_message}\n\n"
            if user_messages:
                user_content += user_messages[0]["content"]
                formatted_messages.append({"role": "user", "content": user_content})
                user_messages = user_messages[1:]
            else:
                formatted_messages.append({"role": "user", "content": user_content})
        
        # Add remaining messages in alternating order
        for i in range(max(len(user_messages), len(assistant_messages))):
            if i < len(assistant_messages):
                formatted_messages.append({"role": "assistant", "content": assistant_messages[i]["content"]})
            if i < len(user_messages):
                formatted_messages.append({"role": "user", "content": user_messages[i]["content"]})
        
        return formatted_messages
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """Generate completion using Anthropic API"""
        try:
            # Convert messages to Anthropic format
            formatted_messages = self._format_messages(messages)
            
            # Prepare the API request payload
            payload = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": 4096
            }
            
            # Handle JSON response format if requested
            if response_format and response_format.get("type") == "json_object":
                # Tell Claude to respond with JSON
                last_message = formatted_messages[-1]
                if isinstance(last_message["content"], str):  # Simple content
                    last_message["content"] += "\n\nPlease format your response as a valid JSON object."
                else:  # Content is a list or other complex type
                    # Handle appropriately based on the structure
                    pass
            
            # Make the API request
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from Anthropic's response format
            return result["content"][0]["text"]
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    def generate_streaming_completion(self, messages, temperature=0.7, response_format=None):
        """Generate streaming completion using Anthropic API"""
        try:
            # Convert messages to Anthropic format
            formatted_messages = self._format_messages(messages)
            
            # Prepare the API request payload with streaming enabled
            payload = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": 4096,
                "stream": True  # Enable streaming
            }
            
            # Handle JSON response format if requested
            if response_format and response_format.get("type") == "json_object":
                # Tell Claude to respond with JSON
                last_message = formatted_messages[-1]
                if isinstance(last_message["content"], str):  # Simple content
                    last_message["content"] += "\n\nPlease format your response as a valid JSON object."
            
            # Set up the headers
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Make the streaming API request
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                stream=True  # Enable HTTP streaming
            )
            
            response.raise_for_status()
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Skip empty lines and parse the event data
                    line = line.decode('utf-8')
                    
                    # Claude's API sends "data: {...}" prefixed lines
                    if line.startswith('data: '):
                        data = line[6:]  # Remove the "data: " prefix
                        
                        # Skip event end marker
                        if data == "[DONE]":
                            continue
                        
                        try:
                            # Parse JSON data from the event
                            result = json.loads(data)
                            
                            # Extract content delta from the event if it exists
                            if (
                                result.get("type") == "content_block_delta" and 
                                "delta" in result and 
                                "text" in result["delta"]
                            ):
                                yield result["delta"]["text"]
                        except json.JSONDecodeError:
                            # Handle any JSON parsing errors
                            continue
                            
        except Exception as e:
            yield f"Anthropic API streaming error: {str(e)}"


class MistralClient(ModelClient):
    """Client for Mistral AI models"""
    def __init__(self, model_name="mistral-large-latest"):
        super().__init__(model_name)
        
        # Get API key using our utility function
        self.api_key = get_api_key("mistral")
            
        if not self.api_key:
            raise ValueError("Mistral API key not found. Please provide it in the sidebar settings.")
        
        # Default API endpoint
        self.api_endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """Generate completion using Mistral API"""
        try:
            # Prepare the API request payload - Mistral format is similar to OpenAI
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Handle JSON response format if requested
            if response_format and response_format.get("type") == "json_object":
                # Add JSON instruction to the last message
                last_message = messages[-1]
                if isinstance(last_message["content"], str):
                    last_message["content"] += "\n\nFormat your response as a valid JSON object."
            
            # Make the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from Mistral's response format (similar to OpenAI)
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"Mistral API error: {str(e)}")


class OllamaClient(ModelClient):
    """Client for Ollama local models"""
    def __init__(self, model_name="llama3"):
        super().__init__(model_name)
        
        # Use default Ollama URL or get from environment
        self.ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Check if Ollama is available
        try:
            # Test connection to Ollama service
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            
            # Store available models
            self.available_models = [model["name"] for model in response.json().get("models", [])]
            
            if self.model_name not in self.available_models and self.available_models:
                # If requested model is not available but others are, use the first available
                self.model_name = self.available_models[0]
                
                # Log warning about model switch
                logging.warning(f"Requested model '{model_name}' not available, using '{self.model_name}' instead")
                
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama service at {self.ollama_url}: {str(e)}")
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """Generate completion using Ollama API"""
        try:
            # Handle JSON response format by adding instruction to the prompt
            if response_format and response_format.get("type") == "json_object":
                last_message = messages[-1]
                if isinstance(last_message["content"], str):
                    last_message["content"] += "\n\nRespond only with a valid JSON object."
            
            # Call Ollama API to generate completion
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                }
            )
            
            # Extract the assistant's response
            return response["message"]["content"]
            
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
            
    def generate_streaming_completion(self, messages, temperature=0.7, response_format=None):
        """Generate streaming completion using Ollama API"""
        try:
            # Handle JSON response format if requested
            if response_format and response_format.get("type") == "json_object":
                last_message = messages[-1]
                if isinstance(last_message["content"], str):
                    last_message["content"] += "\n\nRespond only with a valid JSON object."
            
            # Stream response from Ollama
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                },
                stream=True  # Enable streaming
            )
            
            # Yield chunks as they arrive
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                    
        except Exception as e:
            yield f"Ollama API streaming error: {str(e)}"


class CustomLLMClient(ModelClient):
    """Client for custom LLM API endpoints"""
    def __init__(self, model_name="custom-model"):
        super().__init__(model_name)
        
        # Get API endpoint and key from environment or session state
        self.api_endpoint = get_api_key("custom_api_endpoint") or st.session_state.get('custom_api_endpoint', "")
        self.api_key = get_api_key("custom_api_key") or st.session_state.get('custom_api_key', "")
        
        if not self.api_endpoint:
            raise ValueError("Custom LLM API endpoint not configured. Please provide it in the sidebar settings.")
    
    def generate_completion(self, messages, temperature=0.7, response_format=None):
        """Generate completion using custom LLM API"""
        try:
            # Create headers with authorization if API key is available
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Prepare payload - assumes OpenAI-compatible API format
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Add response format if specified
            if response_format:
                payload["response_format"] = response_format
            
            # Make the API request
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Try to extract content based on OpenAI format first
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    return result["choices"][0]["message"]["content"]
            
            # Try Anthropic format
            if "content" in result and len(result["content"]) > 0:
                if "text" in result["content"][0]:
                    return result["content"][0]["text"]
            
            # If we can't parse the response format, return the raw JSON
            return json.dumps(result)
            
        except Exception as e:
            raise Exception(f"Custom LLM API error: {str(e)}")
            
    def generate_streaming_completion(self, messages, temperature=0.7, response_format=None):
        """Generate streaming completion using custom LLM API"""
        try:
            # Create headers with authorization if API key is available
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Prepare payload - assumes OpenAI-compatible API format
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True  # Enable streaming
            }
            
            # Add response format if specified
            if response_format:
                payload["response_format"] = response_format
            
            # Make the streaming API request
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            
            # Process the streaming response - try to handle both OpenAI and Anthropic formats
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    
                    # Check if it's Server-Sent Events format (OpenAI, Anthropic)
                    if line.startswith('data: '):
                        data = line[6:]  # Remove the "data: " prefix
                        
                        # Skip event end marker
                        if data == "[DONE]":
                            continue
                        
                        try:
                            # Parse JSON data from the event
                            result = json.loads(data)
                            
                            # Try OpenAI format first
                            if "choices" in result and len(result["choices"]) > 0:
                                if "delta" in result["choices"][0] and "content" in result["choices"][0]["delta"]:
                                    content = result["choices"][0]["delta"]["content"]
                                    if content:  # Only yield non-empty content
                                        yield content
                                        continue
                            
                            # Try Anthropic format
                            if "type" in result and result.get("type") == "content_block_delta":
                                if "delta" in result and "text" in result["delta"]:
                                    yield result["delta"]["text"]
                                    continue
                            
                            # If we don't recognize the format but there might be content
                            if "content" in result:
                                yield str(result["content"])
                                continue
                                
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                    else:
                        # Try to parse as JSON directly (non-SSE format)
                        try:
                            result = json.loads(line)
                            if "content" in result:
                                yield str(result["content"])
                        except json.JSONDecodeError:
                            # If it's not JSON, treat as plain text
                            yield line
                            
        except Exception as e:
            yield f"Custom LLM API streaming error: {str(e)}"


def get_model_client(provider, model_name=None):
    """
    Factory function to get an appropriate model client.
    
    Args:
        provider: Name of the provider (openai, anthropic, mistral, ollama, custom)
        model_name: Optional specific model name
    
    Returns:
        A ModelClient instance
    """
    provider = provider.lower()
    
    # Default models for each provider
    default_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "mistral": "mistral-large-latest",
        "ollama": "llama3",  # Default Ollama model
        "custom": "custom-model"  # Default Custom LLM model
    }
    
    # Use default model if none specified
    if not model_name:
        model_name = default_models.get(provider)
    
    # If we still don't have a model name or provider is unsupported, raise error
    if model_name is None:
        raise ValueError(f"No default model for provider: {provider}")
    
    if provider == "openai":
        return OpenAIClient(model_name)
    elif provider == "anthropic":
        return AnthropicClient(model_name)
    elif provider == "mistral":
        return MistralClient(model_name)
    elif provider == "ollama":
        return OllamaClient(model_name)
    elif provider == "custom":
        return CustomLLMClient(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def list_available_models(provider=None):
    """
    List available models for a provider or all providers.
    
    Args:
        provider: Optional provider name to filter results
    
    Returns:
        Dictionary of providers and their available models
    """
    # Static model lists for cloud providers
    models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "mistral": ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"],
        "ollama": ["llama3", "llama3:8b", "mistral", "codellama", "phi3", "gemma", "llava", "mixtral"], # Default models
        "custom": ["custom-model"] # Default custom model
    }
    
    # Try to get actual Ollama models from the server if available
    try:
        if provider is None or provider.lower() == "ollama":
            # Use default Ollama URL or get from environment
            ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags")
            
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if available_models:
                    # Update with actual available models
                    models["ollama"] = available_models
    except Exception as e:
        # If we can't connect to Ollama, just use the default list
        logging.warning(f"Could not retrieve Ollama models: {str(e)}")
    
    if provider:
        provider = provider.lower()
        if provider in models:
            return {provider: models[provider]}
        else:
            return {}
    
    return models

"""
Example usage:

# For OpenAI
openai_client = get_model_client("openai", "gpt-4o")
response = openai_client.generate_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
)

# For Anthropic
anthropic_client = get_model_client("anthropic", "claude-3-5-sonnet-20241022")
response = anthropic_client.generate_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
)

# For Mistral
mistral_client = get_model_client("mistral", "mistral-large-latest")
response = mistral_client.generate_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
)

# For Ollama (local models)
ollama_client = get_model_client("ollama", "llama3")
response = ollama_client.generate_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
)

# For Custom LLM API
# Note: Requires setting custom_api_endpoint and optionally custom_api_key in session state or .env
custom_client = get_model_client("custom", "custom-model")
response = custom_client.generate_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
)

# For streaming responses
for chunk in ollama_client.generate_streaming_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.7
):
    print(chunk, end="", flush=True)
"""