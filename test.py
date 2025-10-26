"""
Bank Statement Processing API
Processes ZIP files containing PDF bank statements using Langgraph and HuggingFace LLM
"""

import os
import zipfile
import tempfile
import json
import re
from typing import List, Dict, Any, TypedDict, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEndpoint

# Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "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")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

def validate_huggingface_token():
    """Validate HuggingFace token and model availability"""
    if not HUGGINGFACE_API_TOKEN:
        print("ERROR: HUGGINGFACE_API_TOKEN not found!")
        print("Please set your HuggingFace API token in environment variables or .env file")
        return False
    
    if not HUGGINGFACE_API_TOKEN.startswith('hf_'):
        print("WARNING: HuggingFace token doesn't start with 'hf_' - this might be invalid")
    
    print(f"HuggingFace Token loaded: {HUGGINGFACE_API_TOKEN[:15]}...")
    print(f"Using model: {LLM_MODEL}")
    
    # Test LLM initialization
    try:
        test_llm = get_llm()
        print("✓ LLM initialization successful")
        return True
    except Exception as e:
        print(f"LLM initialization failed: {e}")
        print("This could be due to:")
        print("1. Invalid API token")
        print("2. Model not available")
        print("3. Network issues")
        print("4. Insufficient API credits")
        return False

# Validate setup on startup
if not validate_huggingface_token():
    print("\nWARNING: HuggingFace setup validation failed!")
    print("The API will still start, but LLM extraction may not work.")
    print("Regex fallback will be used if LLM fails.\n")

app = FastAPI(title="Bank Statement Processor API")

# Response Models
class StatementData(BaseModel):
    statement_period: str
    total_credits: float
    total_debits: float

class FileResult(BaseModel):
    file_name: str
    status: str
    data: Dict[str, Any] | None = None
    error: str | None = None

class ProcessingResult(BaseModel):
    summary: Dict[str, Any]
    files: List[FileResult]

# Langgraph State
class GraphState(TypedDict):
    zip_path: str
    pdf_files: List[str]
    results: Annotated[List[Dict[str, Any]], operator.add]
    total_credits: float
    total_debits: float
    files_processed: int
    files_failed: int

def get_llm():
    """Initialize HuggingFace LLM"""
    try:
        return HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("This might be due to:")
        print("1. Invalid HuggingFace API token")
        print("2. Model not available")
        print("3. Network connectivity issues")
        raise e

def unzip_file(state: GraphState) -> GraphState:
    """Extract all PDF files from ZIP archive"""
    zip_path = state["zip_path"]
    pdf_files = []
    
    try:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all PDFs
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        state["pdf_files"] = pdf_files
        state["results"] = []
        state["total_credits"] = 0.0
        state["total_debits"] = 0.0
        state["files_processed"] = 0
        state["files_failed"] = 0
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to unzip file: {str(e)}")
    
    return state

def extract_with_regex(text: str) -> Dict[str, Any] | None:
    """Fallback regex-based extraction when LLM fails"""
    try:
        # Extract period/month from text
        period_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', text)
        statement_period = period_match.group() if period_match else "Unknown Period"
        
        # Extract amounts with Credit/Debit labels
        credit_pattern = r'Credit\s+MYR\s+([\d,]+\.?\d*)'
        debit_pattern = r'Debit\s+MYR\s+([\d,]+\.?\d*)'
        
        credit_matches = re.findall(credit_pattern, text)
        debit_matches = re.findall(debit_pattern, text)
        
        # Convert to floats and sum
        total_credits = sum(float(amount.replace(',', '')) for amount in credit_matches)
        total_debits = sum(float(amount.replace(',', '')) for amount in debit_matches)
        
        if total_credits > 0 or total_debits > 0:
            return {
                "statement_period": statement_period,
                "total_credits": total_credits,
                "total_debits": total_debits
            }
        
        return None
    except Exception as e:
        print(f"Regex extraction error: {e}")
        return None

def extract_statement_data(text: str) -> Dict[str, Any] | None:
    """Extract financial data using LLM"""
    llm = get_llm()
    
    prompt_text = f"""Extract financial data from this bank statement and return ONLY a JSON object.

Bank Statement:
{text[:3000]}

Extract these fields:
1. statement_period: The period/month (e.g., "September 2025")
2. total_credits: Sum of all Credit amounts (remove MYR, add all credits)
3. total_debits: Sum of all Debit amounts (remove MYR, add all debits)

Return format (ONLY JSON, nothing else):
{{"statement_period": "September 2025", "total_credits": 1234.56, "total_debits": 789.00}}

JSON Output:"""
    
    try:
        # For conversational task, wrap in a list of messages
        response = llm.invoke([{"role": "user", "content": prompt_text}])
        
        # Extract content from conversational response
        if isinstance(response, str):
            content = response
        elif isinstance(response, list) and len(response) > 0:
            # Conversational format returns list of messages
            if isinstance(response[0], dict) and 'content' in response[0]:
                content = response[0]['content']
            else:
                content = str(response[0])
        elif hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'generated_text'):
            content = response.generated_text
        else:
            content = str(response)
        
        print(f"LLM Response: {content[:500]}")
        
        # Parse JSON from response - try multiple approaches
        result = None
        
        # Method 1: Look for JSON object with statement_period
        json_match = re.search(r'\{[^{}]*"statement_period"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Method 2: Look for any JSON object in the response
        if not result:
            json_matches = re.findall(r'\{[^{}]*\}', content)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if "statement_period" in parsed or "total_credits" in parsed or "total_debits" in parsed:
                        result = parsed
                        break
                except json.JSONDecodeError:
                    continue
        
        # Method 3: Try parsing the entire content
        if not result:
            try:
                result = json.loads(content.strip())
            except json.JSONDecodeError:
                pass
        
        print(f"Parsed JSON: {result}")
        
        # Validate required fields
        if result and all(k in result for k in ["statement_period", "total_credits", "total_debits"]):
            result["total_credits"] = float(result["total_credits"]) if result["total_credits"] is not None else 0.0
            result["total_debits"] = float(result["total_debits"]) if result["total_debits"] is not None else 0.0
            
            if result["total_credits"] > 0 or result["total_debits"] > 0:
                return result
        
        # Fallback: Try regex-based extraction if LLM failed
        print("LLM extraction failed, trying regex fallback...")
        return extract_with_regex(text)
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
        print(f"Text sample: {text[:300]}")
        print("Trying regex fallback...")
        return extract_with_regex(text)

def process_pdf(state: GraphState) -> GraphState:
    """Process each PDF using LLM extraction"""
    pdf_files = state["pdf_files"]
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        
        try:
            # Extract text from PDF
            reader = PdfReader(pdf_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            print(f"\n{'='*60}")
            print(f"Processing: {file_name}")
            print(f"Extracted text length: {len(text_content)} chars")
            print(f"Text preview: {text_content[:300]}")
            print(f"{'='*60}\n")
            
            # Extract data using LLM
            extracted_data = extract_statement_data(text_content)
            
            if extracted_data:
                result = {
                    "file_name": file_name,
                    "status": "processed",
                    "data": extracted_data
                }
                
                # Update totals
                state["total_credits"] += extracted_data.get("total_credits", 0.0)
                state["total_debits"] += extracted_data.get("total_debits", 0.0)
                state["files_processed"] += 1
                
                print(f"  Successfully processed {file_name}")
                print(f"  Credits: {extracted_data.get('total_credits', 0.0)}")
                print(f"  Debits: {extracted_data.get('total_debits', 0.0)}")
            else:
                result = {
                    "file_name": file_name,
                    "status": "failed",
                    "error": "Could not extract required fields."
                }
                state["files_failed"] += 1
                print(f"✗ Failed to extract data from {file_name}")
            
            state["results"].append(result)
            
        except Exception as e:
            result = {
                "file_name": file_name,
                "status": "failed",
                "error": f"Processing error: {str(e)}"
            }
            state["results"].append(result)
            state["files_failed"] += 1
            print(f" Error processing {file_name}: {str(e)}")
    
    return state

def build_workflow():
    """Build Langgraph workflow"""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("unzip", unzip_file)
    workflow.add_node("process", process_pdf)
    
    workflow.set_entry_point("unzip")
    workflow.add_edge("unzip", "process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

@app.post("/process-statements", response_model=ProcessingResult)
async def process_statements(file: UploadFile = File(...)):
    """Main endpoint to process ZIP file containing PDF bank statements"""
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Initialize state
        initial_state = GraphState(
            zip_path=tmp_path,
            pdf_files=[],
            results=[],
            total_credits=0.0,
            total_debits=0.0,
            files_processed=0,
            files_failed=0
        )
        
        # Run workflow
        workflow = build_workflow()
        final_state = workflow.invoke(initial_state)
        
        # Build response
        response = ProcessingResult(
            summary={
                "grand_total_credits": round(final_state["total_credits"], 2),
                "grand_total_debits": round(final_state["total_debits"], 2),
                "total_files_processed": final_state["files_processed"],
                "total_files_failed": final_state["files_failed"]
            },
            files=final_state["results"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": LLM_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)