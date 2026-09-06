import subprocess
import sys

packages = [
    "langchain",
    "langchain-community",
    "langchain-huggingface",
    "sentence-transformers",
    "faiss-cpu",
    "pypdf",
    "docx2txt"
]

print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip"] + packages)
print("All packages installed successfully!")