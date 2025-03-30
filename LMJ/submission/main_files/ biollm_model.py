import os
import requests
from aixplain.factories import ModelFactory
from typing import Dict, Any, Optional, Union
import logging

class BioLLM:
    """
    Biomedical processing system with dual-mode RAG support:
    - Automatic query generation from input
    - Manual query specification
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["TEAM_API_KEY"] = api_key
        elif "TEAM_API_KEY" not in os.environ:
            raise ValueError("API key required as argument or environment variable")
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model IDs
        self.bio_llm_id = "67ddc4b1181c58b7238eb33e"#"677c18696eb5634c19191911"
        self.translate_id = "66a7e086f12784226d54d4a7"
        self.speech_recognition_id = "6610617ff1278441b6482530"
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize AI models with error handling"""
        try:
            self.bio_llm = ModelFactory.get(self.bio_llm_id)
            self.translator = ModelFactory.get(self.translate_id)
            self.speech_recognizer = ModelFactory.get(self.speech_recognition_id)
            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise RuntimeError("Model initialization error") from e

    def process_pipeline(
        self,
        input_type: str = "text",
        text: Optional[str] = None,
        source_language: Optional[str] = None,
        audio_path: Optional[str] = None,
        rag_query: Optional[str] = None,
        rag_category: Optional[str] = None,
        target_language: str = "en",
        bio_llm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main processing pipeline with dual RAG modes
        Args:
            rag_query: Optional explicit search query
            rag_category: Required category filter for RAG
        """
        response = {
            "status": "processing",
            "steps": {},
            "errors": []
        }

        try:
            # Input validation
            if input_type not in ["text", "audio"]:
                raise ValueError("Invalid input_type. Use 'text' or 'audio'")

            # Process input
            input_data = self._process_input(
                input_type=input_type,
                text=text,
                audio_path=audio_path,
                source_language=source_language
            )
            response["steps"]["input_processing"] = input_data
            
            if input_data["status"] != "success":
                raise ValueError("Input processing failed")

            # Translation
            translated = self._handle_translation(
                input_data=input_data,
                target_language=target_language
            )
            response["steps"]["translation"] = translated
            
            # RAG processing
            rag_data = None
            if rag_category:
                final_query = rag_query or translated["translated_text"]
                rag_data = self._handle_rag(
                    query=final_query,
                    category=rag_category
                )
                response["steps"]["rag_processing"] = rag_data

            # BioLLM processing
            final_result = self._process_with_biollm(
                translated_text=translated["translated_text"],
                rag_data=rag_data,
                params=bio_llm_params or {}
            )
            response["steps"]["bio_llm_processing"] = final_result

            response.update({
                "status": "success",
                "final_result": final_result["result"],
                "translated_text": translated["translated_text"],
                "rag_data": rag_data["rag_result"] if rag_data else None
            })

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            response.update({
                "status": "error",
                "message": str(e),
                "errors": response.get("errors", []) + [str(e)]
            })

        return response

    def _process_input(
        self,
        input_type: str,
        text: Optional[str],
        audio_path: Optional[str],
        source_language: Optional[str]
    ) -> Dict[str, Any]:
        """Process input with validation"""
        if input_type == "text":
            return self._process_text(text, source_language)
        return self._process_audio(audio_path, source_language)

    def _process_text(
        self,
        text: str,
        source_language: str
    ) -> Dict[str, Any]:
        """Process text input"""
        if not text:
            raise ValueError("Text input requires 'text' parameter")
        if not source_language:
            raise ValueError("Source language required for text input")
        
        return {
            "text": text,
            "source_language": source_language,
            "status": "success",
            "message": "Text processed"
        }

    def _process_audio(
        self,
        audio_path: str,
        source_language: Optional[str]
    ) -> Dict[str, Any]:
        """Process audio input with speech recognition"""
        if not audio_path:
            raise ValueError("Audio input requires 'audio_path'")
        
        try:
            result = self.speech_recognizer.run({"source_audio": audio_path})
            return self._parse_speech_result(result, source_language)
        except Exception as e:
            return {
                "text": "",
                "source_language": source_language or "en",
                "status": "error",
                "message": f"Speech recognition failed: {str(e)}"
            }

    def _parse_speech_result(
        self,
        result: Any,
        fallback_lang: Optional[str]
    ) -> Dict[str, Any]:
        """Parse speech recognition result"""
        try:
            if hasattr(result, 'data'):
                data = result.data
                text = data.get('text', '') if isinstance(data, dict) else str(data)
                lang = data.get('language', fallback_lang) if isinstance(data, dict) else fallback_lang
            else:
                text = str(result)
                lang = fallback_lang

            return {
                "text": text,
                "source_language": lang or "en",
                "status": "success",
                "message": "Audio processed"
            }
        except Exception as e:
            self.logger.error(f"Speech parse error: {e}")
            return {
                "text": "",
                "source_language": fallback_lang or "en",
                "status": "error",
                "message": f"Parse failed: {str(e)}"
            }

    def _handle_translation(
        self,
        input_data: Dict[str, Any],
        target_language: str
    ) -> Dict[str, Any]:
        """Handle translation process"""
        source_lang = input_data.get("source_language", "en")
        text = input_data.get("text", "")

        if source_lang == target_language or not text:
            return {
                "translated_text": text,
                "status": "success",
                "message": "No translation needed"
            }

        try:
            result = self.translator.run({
                "text": text,
                "sourcelanguage": source_lang,
                "targetlanguage": target_language
            })
            translated = result.data if hasattr(result, 'data') else text
            return {
                "translated_text": translated,
                "status": "success",
                "message": "Translated successfully"
            }
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return {
                "translated_text": text,
                "status": "error",
                "message": f"Translation failed: {str(e)}"
            }

    def _handle_rag(
        self,
        query: str,
        category: str
    ) -> Dict[str, Any]:
        """Execute RAG search with query"""
        if not query:
            return {
                "rag_result": "",
                "status": "error",
                "message": "Empty query"
            }

        try:
            response = requests.post(
                "https://models.aixplain.com/api/v2/execute/67dee7599bd803001dba3ca8",
                headers={
                    "x-api-key": os.environ["TEAM_API_KEY"],
                    "Content-Type": "application/json"
                },
                json={
                    "action": "search",
                    "data": query,
                    "payload": {
                        "filters": {
                            "field": "meta.attributes.category",
                            "operator": "==",
                            "value": category
                        }
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "SUCCESS" and result.get("completed"):
                return {
                    "rag_result": result.get("data", ""),
                    "status": "success"
                }
            
            return {
                "rag_result": "",
                "status": "error",
                "message": f"RAG failed: {result.get('status', 'UNKNOWN')}"
            }

        except Exception as e:
            self.logger.error(f"RAG error: {e}")
            return {
                "rag_result": "",
                "status": "error",
                "message": f"RAG request failed: {str(e)}"
            }

    def _process_with_biollm(
        self,
        translated_text: str,
        rag_data: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final response with BioLLM"""
        try:
            context = params.get("context", self._default_context())
            combined_input = translated_text
            
            if rag_data and rag_data.get("rag_result"):
                combined_input += f"\n\nMedical Context: {rag_data['rag_result']}"

            result = self.bio_llm.run({
                "data": combined_input,
                "temperature": str(params.get("temperature", 1.0)),
                "top_p": str(params.get("top_p", 0.9)),
                "top_k": str(params.get("top_k", 50)),
                "max_tokens": str(params.get("max_tokens", 100)),
                "context": "You are an expert and experienced from the healthcare and biomedical domain with extensive "
                      "medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed "
                      "by Saama AI Labs. who's willing to help answer the user's query with explanation. In your "
                      "explanation, leverage your deep medical expertise such as relevant anatomical structures, "
                      "physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical "
                      "concepts. Use precise medical terminology while still aiming to make the explanation clear "
                      "and accessible to a general audience."
            })

            return {
                "result": result.data if hasattr(result, 'data') else str(result),
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"BioLLM error: {e}")
            return {
                "result": None,
                "status": "error",
                "message": f"BioLLM failed: {str(e)}"
            }

    def _default_context(self) -> str:
        """Default medical context"""
        return (
            "As OpenBioLLM, provide professional medical analysis considering: "
            "1. Patient information\n2. Clinical context\n3. Current guidelines\n"
            "Include:\n- Differential diagnosis\n- Recommended tests\n- Treatment options\n"
            "Use medical terminology while maintaining clarity."
        )
