from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tempfile
from model import BioLLM  # Import your BioLLM class

# Initialize Flask app
current_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=current_dir)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize BioLLM with your API key
API_KEY = "TEAM_API_KEY"  # Better to use environment variable
bio_llm = BioLLM(api_key=API_KEY)

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        print(f"Current directory: {current_dir}")
        print(f"Looking for index.html in: {current_dir}")

        # Check if file exists
        file_path = os.path.join(current_dir, 'index.html')
        print(f"Full file path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        return send_from_directory(current_dir, 'index.html')
    except Exception as e:
        app.logger.error(f"Error serving index.html: {str(e)}")
        return f"Error: {str(e)}", 404
        

@app.route('/api/process-input', methods=['POST'])
def process_input():
    """Handle text input from frontend"""
    data = request.json
    text = data.get('text', '')
    source_language = data.get('sourceLanguage', 'en')
    
    try:
        # Process the text using BioLLM pipeline
        result = bio_llm.process_pipeline(
            input_type="text",
            text=text,
            source_language=source_language,
            target_language="en",  # Process in English internally
            rag_query=text,  # Use the same text for RAG query
            rag_category="general"  # Default category, modify as needed
        )
        
        # Extract the final response
        bio_llm_result = result.get("steps", {}).get("bio_llm_processing", {}).get("result", "")
        
        # If original language wasn't English, translate response back
        if source_language.lower() != "en":
            translated_back = bio_llm.translate_text(
                {"text": bio_llm_result, "source_language": "en"}, 
                target_language=source_language
            )
            response_text = translated_back.get("translated_text", bio_llm_result)
        else:
            response_text = bio_llm_result
            
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """Handle audio input from frontend"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    source_language = request.form.get('sourceLanguage', 'en')
    
    try:
        # Save audio file temporarily
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Process the audio using BioLLM pipeline
        result = bio_llm.process_pipeline(
            input_type="speech",
            audio_path=filepath,
            source_language=source_language,
            target_language="en",  # Process in English internally
            rag_category="general"  # Default category, modify as needed
        )
        
        # Get the transcribed text
        input_processing = result.get("steps", {}).get("input_processing", {})
        transcribed_text = input_processing.get("text", "")
        
        # Get the final response
        bio_llm_result = result.get("steps", {}).get("bio_llm_processing", {}).get("result", "")
        
        # If original language wasn't English, translate response back
        if source_language.lower() != "en":
            translated_back = bio_llm.translate_text(
                {"text": bio_llm_result, "source_language": "en"}, 
                target_language=source_language
            )
            response_text = translated_back.get("translated_text", bio_llm_result)
        else:
            response_text = bio_llm_result
        
        # Clean up temp file
        os.remove(filepath)
            
        return jsonify({
            "transcribed": transcribed_text,
            "response": response_text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)