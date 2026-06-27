import os
import sys
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

def print_header(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

def test_environment_variables():
    print_header("1. Checking Environment Variables")
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "COHERE_API_KEY", "GROQ_API_KEY"]
    all_ok = True
    
    for var in required_vars:
        val = os.getenv(var)
        if val:
            # Mask the key for privacy
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            print(f"[OK] {var}: Detected ({masked})")
        else:
            print(f"[MISSING] {var} is not set in your .env file")
            all_ok = False
            
    return all_ok

def test_supabase_connection():
    print_header("2. Testing Supabase Connection")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("[WARNING] Cannot test Supabase because URL or Key is missing.")
        return False
        
    # Normalize URL (remove trailing /v1 or /rest/v1)
    if "/rest/v1" in url:
        url = url.split("/rest/v1")[0]
    elif "/v1" in url:
        url = url.split("/v1")[0]
    url = url.rstrip("/")
        
    try:
        from supabase import create_client
        supabase = create_client(url, key)
        
        # Test querying the attendance table
        print("Connecting and querying 'attendance' table...")
        attendance_resp = supabase.table("attendance").select("count", count="exact").limit(1).execute()
        print(f"[OK] 'attendance' table is reachable! (Row count: {attendance_resp.count if hasattr(attendance_resp, 'count') else 'N/A'})")
        
        # Test querying the metrics table
        print("Connecting and querying 'metrics' table...")
        metrics_resp = supabase.table("metrics").select("count", count="exact").limit(1).execute()
        print(f"[OK] 'metrics' table is reachable! (Row count: {metrics_resp.count if hasattr(metrics_resp, 'count') else 'N/A'})")
        
        return True
    except Exception as e:
        print(f"[ERROR] Supabase Connection Failed: {e}")
        return False

def test_knowledge_base():
    print_header("3. Testing Knowledge Base & FAISS Loading")
    
    # Check if folder exists
    kb_path = "Knowledge Base"
    if not os.path.exists(kb_path):
        print(f"[ERROR] Knowledge Base folder '{kb_path}' not found at: {os.path.abspath(kb_path)}")
        return False
        
    print(f"[OK] Folder '{kb_path}' exists.")
    
    # Check files
    index_faiss = os.path.join(kb_path, "index.faiss")
    index_pkl = os.path.join(kb_path, "index.pkl")
    
    if not os.path.exists(index_faiss) or not os.path.exists(index_pkl):
        print("[ERROR] Missing index.faiss or index.pkl inside Knowledge Base folder!")
        return False
        
    print("[OK] index.faiss and index.pkl files found.")
    
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("[WARNING] Skipping FAISS index load test: COHERE_API_KEY is missing (needed for embeddings initialization).")
        return True # Folders are there, so this counts as structurally OK
        
    try:
        print("Initializing CohereEmbeddings and loading FAISS index...")
        from langchain_cohere import CohereEmbeddings
        from langchain_community.vectorstores import FAISS
        
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_key)
        vector_store = FAISS.load_local(
            kb_path, embeddings, allow_dangerous_deserialization=True
        )
        print("[OK] FAISS index loaded successfully!")
        
        # Test search
        print("Running a mock search query: 'dress code'...")
        results = vector_store.similarity_search("dress code", k=1)
        if results:
            print(f"[OK] Search query returned matching content: '{results[0].page_content[:100]}...'")
        else:
            print("[WARNING] Search query returned no results, but loading succeeded.")
            
        return True
    except Exception as e:
        print(f"[ERROR] FAISS Index Load Failed: {e}")
        return False

def main():
    print("FateAutomata Connection & Knowledge Base Verification")
    
    env_ok = test_environment_variables()
    sb_ok = test_supabase_connection()
    kb_ok = test_knowledge_base()
    
    print_header("Summary")
    if env_ok and sb_ok and kb_ok:
        print("SUCCESS: Everything is ready for Streamlit deployment!")
    else:
        print("SOME CHECKS DID NOT PASS: Please address the issues shown above before deploying.")

if __name__ == "__main__":
    main()
