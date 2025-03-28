import os
from aixplain.factories import ModelFactory
from typing import Dict, Any, Optional, Union
import logging
import requests


class BioLLM:
    """
    A class that integrates various AI models for biomedical text processing with optimized translation handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the BioLLM class with API key and model IDs.
        
        Args:
            api_key: Optional API key for aixplain services
        """
        if api_key:
            os.environ["TEAM_API_KEY"] = api_key
        elif "TEAM_API_KEY" not in os.environ:
            raise ValueError("API key must be provided either as an argument or as an environment variable")
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.bio_llm_id = "67ddc4b1181c58b7238eb33e"
        self.rag_agent_id = "67df1bd1181c58b7238eb7dd"
        self.translate_id = "66a7e086f12784226d54d4a7"
        self.speech_recognition_id = "6610617ff1278441b6482530"
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            self.bio_llm_model = ModelFactory.get(self.bio_llm_id)
            self.rag_agent_model = ModelFactory.get(self.rag_agent_id)
            self.translate_model = ModelFactory.get(self.translate_id)
            self.speech_recognition_model = ModelFactory.get(self.speech_recognition_id)
            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def process_text(self, text: str, source_language: str) -> Dict[str, Any]:
        """Process input text and return structured response"""
        self.logger.info(f"Processing text in {source_language}: {text[:50]}...")
        return {
            "text": text,
            "source_language": source_language,
            "status": "success",
            "message": f"Successfully processed text in {source_language}"
        }
    
    def process_rag_query(self, query: str, category: str) -> Dict[str, Any]:
        """
        Search biomedical data using RAG API endpoint with category filter.
        
        Args:
            query: Search query text
            category: Category filter for document retrieval
            
        Returns:
            Dictionary containing RAG results or error message
        """
        self.logger.info(f"Processing RAG query for category {category}: {query}")
        headers = {
            "x-api-key": os.environ["TEAM_API_KEY"],
            "Content-Type": "application/json"
        }
        payload = {
            "action": "search",
            "data": query,
            "payload": {
                "filters": {
                    "field": "meta.attributes.category",
                    "operator": "==",
                    "value": category
                }
            }
        }
        
        try:
            response = requests.post(
                "https://models.aixplain.com/api/v2/execute/67dee7599bd803001dba3ca8",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "SUCCESS" and result.get("completed"):
                rag_data = result.get("data", "")
                return {"rag_result": rag_data}
            else:
                error_msg = f"RAG query failed with status: {result.get('status', 'UNKNOWN')}"
                self.logger.error(error_msg)
                return {"rag_result": error_msg}
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            self.logger.error(error_msg)
            return {"rag_result": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            return {"rag_result": error_msg}
    
    def speech_recognition(self, audio_path: str) -> Dict[str, Any]:
        """Process audio input and return structured response with detected language"""
        self.logger.info(f"Performing speech recognition on {audio_path}")
        try:
            result = self.speech_recognition_model.run({"source_audio": audio_path})
            
            # Extract text and detected language from response
            if hasattr(result, 'data'):
                recognized_text = result.data.get('text', '')
                detected_language = result.data.get('language', 'en')
            else:
                recognized_text = str(result)
                detected_language = 'en'
            
            return {
                "text": recognized_text,
                "source_language": detected_language,
                "status": "success",
                "message": "Successfully transcribed audio"
            }
        except Exception as e:
            self.logger.error(f"Error in speech recognition: {e}")
            return {
                "text": None,
                "source_language": None,
                "status": "error",
                "message": f"Speech recognition failed: {str(e)}"
            }
    
    def translate_text(self, input_data: Dict[str, Any], target_language: str = "en") -> Dict[str, Any]:
        """Translate text and return structured response"""
        text = input_data.get("text", "")
        source_language = input_data.get("source_language", "en")
        
        self.logger.info(f"Translating from {source_language} to {target_language}")
        
        if source_language == target_language:
            return {
                "translated_text": text,
                "status": "success",
                "message": "No translation needed - same language"
            }
        
        try:
            result = self.translate_model.run({
                "text": text,
                "sourcelanguage": source_language,
                "targetlanguage": target_language
            })
            translated_text = result.data if hasattr(result, 'data') else text
            return {
                "translated_text": translated_text,
                "status": "success",
                "message": f"Successfully translated from {source_language} to {target_language}"
            }
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            return {
                "translated_text": text,
                "status": "error",
                "message": f"Translation failed: {str(e)}"
            }
    
    def process_with_bio_llm(self, 
                            text_data: Dict[str, Any], 
                            rag_data: Optional[Dict[str, Any]] = None,
                            temperature: float = 1.0,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            max_tokens: int = 100,
                            context: Optional[str] = None) -> Dict[str, Any]:
        """Process with BioLLM and return structured response"""
        translated_text = text_data.get("translated_text", "")
        
        rag_context = ""
        if rag_data and "rag_result" in rag_data:
            rag_result = rag_data["rag_result"]
            if isinstance(rag_result, dict):
                rag_context = rag_result.get('data', '')
            else:
                rag_context = str(rag_result)
        
        combined_input = f"{translated_text}\n\nAdditional context: {rag_context}" if rag_context else translated_text
        
        self.logger.info(f"Processing with BioLLM using text: {combined_input[:100]}...")
        
        if not context:
            context = ("You are an expert and experienced from the healthcare and biomedical domain with extensive "
                      "medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed "
                      "by Saama AI Labs. who's willing to help answer the user's query with explanation. In your "
                      "explanation, leverage your deep medical expertise such as relevant anatomical structures, "
                      "physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical "
                      "concepts. Use precise medical terminology while still aiming to make the explanation clear "
                      "and accessible to a general audience.")
        
        try:
            result = self.bio_llm_model.run({
                "data": combined_input,
                "temperature": str(temperature),
                "top_p": str(top_p),
                "top_k": str(top_k),
                "max_tokens": str(max_tokens),
                "context": context
            })
            return {
                "result": result.data if hasattr(result, 'data') else str(result),
                "status": "success",
                "message": "Successfully processed with BioLLM"
            }
        except Exception as e:
            self.logger.error(f"Error in BioLLM processing: {e}")
            return {
                "result": None,
                "status": "error", 
                "message": f"BioLLM processing failed: {str(e)}"
            }
    
    def process_pipeline(self, 
                        input_type: str = "text",
                        text: Optional[str] = None,
                        source_language: Optional[str] = None,
                        audio_path: Optional[str] = None,
                        rag_query: Optional[str] = None,
                        rag_category: Optional[str] = None,
                        target_language: str = "en",
                        bio_llm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main pipeline to process inputs with optimized translation handling"""
        response = {
            "status": "processing",
            "steps": {}
        }
        
        try:
            # Input processing
            if input_type == "text":
                if not text:
                    raise ValueError("Text input requires 'text' parameter")
                if not source_language:
                    raise ValueError("Source language must be provided for text input")
                input_data = self.process_text(text, source_language)
            elif input_type == "speech":
                if not audio_path:
                    raise ValueError("Speech input requires 'audio_path' parameter")
                input_data = self.speech_recognition(audio_path)
                # Update source_language from speech recognition result
                source_language = input_data.get("source_language", "en")
            else:
                raise ValueError("Invalid input type. Choose 'text' or 'speech'")
            
            response["steps"]["input_processing"] = input_data
            
            # Determine source language
            source_lang = input_data.get("source_language", "en")
            
            # Translation handling
            if source_lang != "en":
                translated_data = self.translate_text(input_data, target_language)
            else:
                translated_data = {
                    "translated_text": input_data.get("text", ""),
                    "status": "success",
                    "message": "No translation needed - source language is English"
                }
            response["steps"]["translation"] = translated_data
            
            # RAG processing
            rag_data = None
            if rag_query and rag_category:
                rag_data = self.process_rag_query(rag_query, rag_category)
                response["steps"]["rag_processing"] = rag_data
            
            # BioLLM processing
            if bio_llm_params is None:
                bio_llm_params = {}
                
            result = self.process_with_bio_llm(translated_data, rag_data, **bio_llm_params)
            response["steps"]["bio_llm_processing"] = result
            
            response["status"] = "success"
            response["message"] = "Pipeline completed successfully"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            response["status"] = "error"
            response["message"] = f"Pipeline failed: {str(e)}"
            return response
