# HackerEarth-AI_agent---21561

## BioLLM Medical Assistant

This project is a **Streamlit-based** application that utilizes BioLLM for medical analysis. It supports both **text and audio** inputs for biomedical processing.

## **Setup Instructions**

### **1. Clone the Repository**

```bash
 git clone <your-repo-url>
 cd <your-repo-name>/main_files
```

### **2. Create a Virtual Environment (Recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up the API Key**

The application requires a **TEAM API Key** to function. Export the key using the following command:

```bash
export TEAM_API_KEY="your-team-api-key"
```

If you are using Windows PowerShell:

```powershell
$env:TEAM_API_KEY="your-team-api-key"
```
## **Model Configuration**

To optimize the cost of running the custom model, scale down the instance when not in use. At the start of your session, follow this process:

- Run the custom model until you get a response. This may take up to **7 minutes**.
- Send **3-4 requests**, as the first few may time out during the warming-up phase.
- Once warmed up, the model will be ready for use in your pipeline.

To improve execution speed, modify the **BioLLM model selection** in your code:

```python
# Default: SLM Model (lower cost but slower startup)
self.bio_llm_id = "67ddc4b1181c58b7238eb33e"

# Uncomment the following line for a faster model
# self.bio_llm_id = "677c18696eb5634c19191911"
``` 
### **5. Run the Application**

Execute the following command in **VS Code Terminal**:

```bash
streamlit run /yourdirectory/app.py
```

Replace `/yourdirectory/` with the actual path where **app.py** is located.

### **6. Access the Application**

Once the server starts, open the given **localhost URL** in your web browser to interact with the BioLLM Medical Assistant.

## **File Structure**

```
LMJ/
│── submission/
│    │── main_files/
│    │    │── app.py               # Main Streamlit app
│    │    │── biollm_model.py      # BioLLM model handling
│    │    │── requirements.txt     # Required Python packages
│    │    │── Execution.ipynb      # Jupyter notebook for testing
│    │    │── .env                 # Environment variables (not tracked in Git)
│    │    │── recorder.png         # UI asset
│    │    │── user-profile.png     # UI asset
│    │── README.md                 # This file
```

## **Troubleshooting**

- If `ModuleNotFoundError` occurs, ensure you have activated your virtual environment and installed dependencies.
- If API issues arise, double-check that `TEAM_API_KEY` is correctly exported.
- If Streamlit does not launch, confirm that you are running the command from the correct directory.

---



https://github.com/user-attachments/assets/df54fa26-e129-445d-919d-7f77992a7612

