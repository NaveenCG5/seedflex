
Bank Statement Processing API
**Repository:** naveencg5/seedflex  
**Main file:** test.py

Description
API to process ZIP files containing PDF bank statements. Extracts statement period, total credits, and total debits using a HuggingFace LLM, with a regex fallback for unsupported or corrupted files.

Features
- Upload ZIP files containing multiple PDF statements.  
- Extract financial data (credits, debits, statement period).  
- Automatic fallback extraction using regex if LLM fails.  
- Returns structured JSON summary per file and overall totals.  
- Health check endpoint to verify API and model status.  

Requirements
- Python 3.10+  
- HuggingFace API token (`HUGGINGFACE_API_TOKEN`) in environment variables or `.env` file.  
- Dependencies listed in `requirements.txt`.  

Installation
```bash
git clone https://github.com/naveencg5/seedflex.git
cd seedflex
pip install -r requirements.txt
```

Usage with Postman
1. Ensure the API is running locally (for example using `uvicorn test:app --host 0.0.0.0 --port 8000`).
2. Open Postman and create a **POST** request:

   * URL: `http://0.0.0.0:8000/process-statements`
   * Body → form-data → Key: `file` → Type: `File` → Select your ZIP file containing PDF statements.
3. Send the request.
4. The API will respond with a JSON summary including per-file results and grand totals.

Health Check
* Method: **GET**
* URL: `http://0.0.0.0:8000/health`
* Response includes API status and LLM model being used.

Example Response
```json
{
  "summary": {
    "grand_total_credits": 15750.75,
    "grand_total_debits": 8320.10,
    "total_files_processed": 3,
    "total_files_failed": 1
  },
  "files": [
    {
      "file_name": "statement-jan.pdf",
      "status": "processed",
      "data": {
        "statement_period": "January 2024",
        "total_credits": 5500.00,
        "total_debits": 2264.09
      }
    },
    {
      "file_name": "corrupted.pdf",
      "status": "failed",
      "error": "Could not extract required fields."
    }
  ]
}
```

Notes

* PDFs must contain selectable text (not just scanned images).
* LLM extraction may fail if API token is invalid or credits are insufficient; regex fallback ensures data is still extracted when possible.
