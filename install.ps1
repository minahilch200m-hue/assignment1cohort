$pythonPath = "C:\Program Files\Python39\python.exe"
$packages = "langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf docx2txt"

& "$pythonPath" -m pip install $packages
Write-Host "Install complete" -ForegroundColor Green