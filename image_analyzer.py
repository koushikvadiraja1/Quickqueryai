"""
Image Analyzer Module
This module provides image analysis capabilities using AI models.
It supports both OpenAI and Anthropic multimodal models for image analysis.
"""

import os
import base64
import io
from typing import Dict, Any, Optional, Union
import streamlit as st
from PIL import Image
import json

# Import our modules
import model_clients
from utils import get_api_key

class ImageAnalyzer:
    """
    Class for analyzing images using AI models.
    Supports multiple model providers for image analysis.
    """
    
    def __init__(self, model_provider="openai", model_name=None):
        """
        Initialize the image analyzer with a model provider and model name.
        
        Args:
            model_provider: Model provider to use (openai, anthropic)
            model_name: Specific model name to use (if None, will use provider default)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        
        # Get model client for image analysis
        self.model_client = model_clients.get_model_client(model_provider, model_name)
        
        # Check that the selected model supports image analysis
        supported_providers = ["openai", "anthropic"]
        if model_provider.lower() not in supported_providers:
            raise ValueError(f"Model provider {model_provider} does not support image analysis. Use one of: {', '.join(supported_providers)}")
    
    def encode_image(self, image: Union[str, Image.Image, bytes]) -> str:
        """
        Encode an image to base64 for API submission.
        
        Args:
            image: Image to encode (can be a file path, PIL Image object, or bytes)
            
        Returns:
            Base64-encoded image string
        """
        if isinstance(image, str):
            # Image is a file path
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            # Image is a PIL Image object
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):
            # Image is already bytes
            return base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError("Image must be a file path, PIL Image, or bytes")
    
    def analyze_image(self, image: Union[str, Image.Image, bytes], 
                     prompt: str = "Analyze this image in detail and describe its key elements, context, and any notable aspects.", 
                     temperature: float = 0.7) -> str:
        """
        Analyze an image using the configured AI model.
        
        Args:
            image: Image to analyze (file path, PIL Image object, or bytes)
            prompt: Text prompt to guide the analysis
            temperature: Creativity parameter for model response
            
        Returns:
            Analysis text from the model
        """
        # Encode the image to base64
        base64_image = self.encode_image(image)
        
        if self.model_provider.lower() == "openai":
            return self._analyze_with_openai(base64_image, prompt, temperature)
        elif self.model_provider.lower() == "anthropic":
            return self._analyze_with_anthropic(base64_image, prompt, temperature)
        else:
            raise ValueError(f"Image analysis not supported for provider: {self.model_provider}")
    
    def _analyze_with_openai(self, base64_image: str, prompt: str, temperature: float) -> str:
        """
        Analyze an image using OpenAI's models.
        
        Args:
            base64_image: Base64-encoded image string
            prompt: Text prompt to guide the analysis
            temperature: Creativity parameter for model response
            
        Returns:
            Analysis text from the model
        """
        try:
            from openai import OpenAI
            
            # Get API key
            api_key = get_api_key("openai")
            if not api_key:
                return "OpenAI API key not found. Please add your API key in the Model Settings tab."
                
            client = OpenAI(api_key=api_key)
            
            # Prepare a helpful message for the case of authentication errors
            auth_error_note = (
                "The API key you provided is invalid or doesn't have access to the required capabilities. "
                "Make sure your OpenAI account has access to the vision API features and GPT-4o model."
            )
            
            # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            try:
                response = client.chat.completions.create(
                    model=self.model_name or "gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=1500,
                )
                # Get the response content
                content = response.choices[0].message.content
                
                # Check for content policy decline messages
                if "I'm sorry, I can't assist with that" in content or "I cannot analyze" in content:
                    return "The AI model declined to process this image due to content policy restrictions. Please try a different image."
                    
                return content
            except Exception as api_error:
                error_message = str(api_error).lower()
                if "authentication" in error_message or "auth" in error_message or "key" in error_message:
                    return f"Authentication error: {auth_error_note}"
                elif "permission" in error_message or "access" in error_message:
                    return f"Permission error: Your API key doesn't have access to vision capabilities. {auth_error_note}"
                elif "content policy" in error_message or "violate" in error_message:
                    return "The image couldn't be analyzed due to content policy restrictions."
                else:
                    return f"Error processing image with OpenAI: {str(api_error)}"
                
        except Exception as e:
            return f"Error setting up OpenAI client: {str(e)}"
    
    def _analyze_with_anthropic(self, base64_image: str, prompt: str, temperature: float) -> str:
        """
        Analyze an image using Anthropic's models.
        
        Args:
            base64_image: Base64-encoded image string
            prompt: Text prompt to guide the analysis
            temperature: Creativity parameter for model response
            
        Returns:
            Analysis text from the model
        """
        try:
            import anthropic
            
            # Get API key
            api_key = get_api_key("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not found. Please provide it in the settings.")
                
            client = anthropic.Anthropic(api_key=api_key)
            
            # The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            # do not change this unless explicitly requested by the user
            message = client.messages.create(
                model=self.model_name or "claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=temperature,
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error analyzing image with Anthropic: {str(e)}"
    
    def extract_text_from_image(self, image: Union[str, Image.Image, bytes], 
                               temperature: float = 0.3) -> str:
        """
        Extract text content from an image (OCR).
        
        Args:
            image: Image containing text to extract
            temperature: Creativity parameter for model response
            
        Returns:
            Extracted text from the image
        """
        try:
            # Encode the image to base64
            base64_image = self.encode_image(image)
            
            # More detailed prompt specific for OCR tasks
            prompt = """
            Please extract all text visible in this image. 
            
            Instructions:
            1. Preserve the original structure and layout of the text
            2. Maintain paragraph breaks, bullet points, and other formatting
            3. Only include text that you can clearly see
            4. If there is text in multiple columns, extract each column separately
            5. If you see any tables, preserve the table structure as best as possible
            
            Your response should only contain the extracted text, nothing else.
            """
            
            if self.model_provider.lower() == "openai":
                result = self._analyze_with_openai(base64_image, prompt, temperature)
                return result
            elif self.model_provider.lower() == "anthropic":
                result = self._analyze_with_anthropic(base64_image, prompt, temperature)
                return result
            else:
                return f"Text extraction not supported for provider: {self.model_provider}"
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"
    
    def analyze_image_with_json_output(self, image: Union[str, Image.Image, bytes], 
                                     analysis_type: str = "general",
                                     temperature: float = 0.3) -> Dict[str, Any]:
        """
        Analyze an image and return structured JSON data.
        
        Args:
            image: Image to analyze
            analysis_type: Type of analysis to perform (general, objects, text, faces, colors)
            temperature: Creativity parameter for model response
            
        Returns:
            Dictionary with structured analysis results
        """
        # Encode the image to base64
        base64_image = self.encode_image(image)
        
        prompts = {
            "general": "Analyze this image and provide a JSON response with the following fields: main_subject, setting, background, atmosphere, colors, estimated_time_period, key_objects, and a brief overall_description.",
            "objects": "Identify all distinct objects in this image and return a JSON array of objects with name, position (top-left, center, etc), approximate_size (small, medium, large), and a brief description field for each object.",
            "text": "Extract all text visible in this image and return it as a JSON object with fields: all_text (containing all extracted text), main_heading (if any), paragraphs (array of distinct paragraphs), and language (detected language of the text).",
            "faces": "Identify any faces/people in this image and return a JSON array with each person having: position, estimated_age_range, gender, expression, clothing_description, and distinguishing_features.",
            "colors": "Analyze the color palette of this image and return a JSON object with: dominant_colors (array of hex codes and color names), color_scheme (warm, cool, neutral, vibrant, etc), color_harmony (complementary, analogous, triadic, etc), and mood_conveyed by the colors."
        }
        
        if analysis_type not in prompts:
            analysis_type = "general"
            
        prompt = prompts[analysis_type] + " Format your response as valid JSON."
        
        try:
            # Get model response
            if self.model_provider.lower() == "openai":
                # For OpenAI we can request JSON directly
                from openai import OpenAI
                
                # Get API key
                api_key = get_api_key("openai")
                if not api_key:
                    raise ValueError("OpenAI API key not found. Please provide it in the settings.")
                    
                client = OpenAI(api_key=api_key)
                
                # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = client.chat.completions.create(
                    model=self.model_name or "gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=1500,
                )
                content = response.choices[0].message.content
                
                # Check for content policy decline messages before trying to parse JSON
                if "I'm sorry, I can't assist with that" in content or "I cannot analyze" in content:
                    return {"error": "The AI model declined to process this image due to content policy restrictions. Please try a different image."}
                
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError:
                    return {"error": f"Could not parse JSON from response: {content}"}
            elif self.model_provider.lower() == "anthropic":
                # For Anthropic we'll need to parse the JSON from the text response
                text_response = self._analyze_with_anthropic(base64_image, prompt, temperature)
                
                # Extract JSON from the response
                # Find the first '{' and the last '}'
                start_idx = text_response.find('{')
                end_idx = text_response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = text_response[start_idx:end_idx+1]
                    try:
                        return json.loads(json_str)
                    except:
                        # Try to fix common JSON formatting issues
                        # Replace single quotes with double quotes
                        json_str = json_str.replace("'", '"')
                        try:
                            return json.loads(json_str)
                        except:
                            raise ValueError(f"Could not parse JSON from response: {text_response}")
                else:
                    raise ValueError(f"No JSON found in response: {text_response}")
        except Exception as e:
            return {"error": f"Error analyzing image: {str(e)}"}


# Function to create a demo UI for image analysis
def image_analyzer_ui():
    """
    Create a Streamlit UI for image analysis.
    This function can be called from the main app to add image analysis functionality.
    """
    st.header("✨ Image Analysis")
    
    # Model selection
    with st.expander("Model Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            model_provider = st.selectbox(
                "Model Provider", 
                ["OpenAI", "Anthropic"],
                index=0,
                key="image_analyzer_provider"
            )
        
        with col2:
            if model_provider.lower() == "openai":
                model_name = st.selectbox(
                    "Model",
                    ["gpt-4o", "gpt-4-vision-preview"],
                    index=0,
                    key="image_analyzer_model_openai"
                )
            else:  # Anthropic
                model_name = st.selectbox(
                    "Model",
                    ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
                    index=0,
                    key="image_analyzer_model_anthropic"
                )
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analysis options
        analysis_type = st.radio(
            "Analysis Type",
            ["General Analysis", "Detect Objects", "Extract Text (OCR)", "Analyze People/Faces", "Color Analysis"],
            index=0,
            horizontal=True,
            key="image_analysis_type"
        )
        
        # Custom prompt option
        custom_prompt = st.text_area(
            "Custom Analysis Prompt (optional)",
            placeholder="Enter a custom prompt or leave blank to use the default prompt for the selected analysis type",
            key="image_custom_prompt"
        )
        
        # Analysis button
        if st.button("Analyze Image", key="analyze_image_button"):
            with st.spinner("Analyzing image..."):
                try:
                    # Check for API key availability first
                    api_key = get_api_key(model_provider.lower())
                    if not api_key:
                        st.error(f"No API key found for {model_provider}. Please add your API key in the Model Settings tab.")
                        return

                    # Initialize image analyzer
                    analyzer = ImageAnalyzer(
                        model_provider=model_provider.lower(),
                        model_name=model_name
                    )
                    
                    # Map analysis type to function
                    analysis_type_map = {
                        "General Analysis": "general",
                        "Detect Objects": "objects",
                        "Extract Text (OCR)": "text",
                        "Analyze People/Faces": "faces",
                        "Color Analysis": "colors"
                    }
                    
                    # If text extraction is selected, use the extract_text_from_image function
                    if analysis_type == "Extract Text (OCR)":
                        if custom_prompt:
                            result = analyzer.analyze_image(image, prompt=custom_prompt)
                            st.subheader("Extracted Text (Custom)")
                            st.write(result)
                        else:
                            result = analyzer.extract_text_from_image(image)
                            st.subheader("Extracted Text")
                            st.text_area("Text Content", result, height=300)
                    else:
                        # For other analysis types, use JSON output
                        if custom_prompt:
                            result = analyzer.analyze_image(image, prompt=custom_prompt)
                            st.subheader("Analysis Results (Custom)")
                            st.write(result)
                        else:
                            analysis_key = analysis_type_map.get(analysis_type, "general")
                            result = analyzer.analyze_image_with_json_output(image, analysis_type=analysis_key)
                            
                            st.subheader("Analysis Results")
                            
                            # Display the results in a nice format
                            if analysis_type == "General Analysis":
                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Main Subject:**")
                                        st.write(result.get("main_subject", "N/A"))
                                        
                                        st.markdown("**Setting:**")
                                        st.write(result.get("setting", "N/A"))
                                        
                                        st.markdown("**Time Period:**")
                                        st.write(result.get("estimated_time_period", "N/A"))
                                    
                                    with col2:
                                        st.markdown("**Key Objects:**")
                                        objects = result.get("key_objects", [])
                                        if isinstance(objects, list):
                                            for obj in objects:
                                                st.write(f"- {obj}")
                                        else:
                                            st.write(objects)
                                        
                                        st.markdown("**Colors:**")
                                        st.write(result.get("colors", "N/A"))
                                    
                                    st.markdown("**Overall Description:**")
                                    st.write(result.get("overall_description", "N/A"))
                            else:
                                # For other analysis types, just show the pretty-printed JSON
                                st.json(result)
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Error during image analysis: {error_msg}")
                    
                    if "I'm sorry, I can't assist with that" in error_msg:
                        st.warning("The AI model declined to process this image due to content policy restrictions. Please try a different image.")
                    elif "API key" in error_msg or "Authentication" in error_msg or "401" in error_msg:
                        st.info("Make sure you have provided a valid API key with access to vision models in the app settings.")
                        st.info("Go to the top-right corner API Settings section to add or update your OpenAI API key.")
                    elif "quota" in error_msg or "billing" in error_msg or "exceeded" in error_msg:
                        st.info("Your API key may have exceeded its quota or doesn't have billing set up for vision model access.")
                    else:
                        st.info("Make sure you have provided the appropriate API key in the app settings and that your image is valid.")


if __name__ == "__main__":
    # If run directly, create a simple app for testing
    st.set_page_config(page_title="Image Analyzer", layout="wide")
    image_analyzer_ui()