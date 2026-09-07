'use client';

import { useState, useEffect, useRef } from "react";
const API_BASE = "http://127.0.0.1:8000";


export default function Home() {
  // --- Document Upload State ---
  const [uploadStatus, setUploadStatus] = useState("Ready to upload"); // "Ready to upload", "Processing...", "Indexed ✓", or "Error..."
  const [uploadError, setUploadError] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  // --- Chat State ---
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [isThinking, setIsThinking] = useState(false);

  // Fetch documents on load
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setUploadStatus("Ready to upload");
      setUploadError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setUploadError("Please select a file first");
      return;
    }

    setUploadStatus("Processing...");
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setUploadStatus("Indexed ✓");
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        await fetchDocuments();
      } else {
        setUploadStatus("Error");
        setUploadError(data.message || "Upload failed");
      }
    } catch (err) {
      setUploadStatus("Error");
      setUploadError("Network error: Unable to reach backend");
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = inputValue.trim();
    if (!text || isThinking) return;

    // Add user message immediately
    setMessages((prev) => [...prev, { role: "user", text }]);
    setInputValue("");
    setIsThinking(true);

    const payload = {
      message: text,
      session_id: sessionId || "",
    };

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (data.session_id) {
        setSessionId(data.session_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer || "No answer returned",
          citations: data.citations || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Error: Unable to connect to backend server.",
          citations: [],
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6 font-sans text-gray-800">
      <div className="max-w-2xl mx-auto space-y-6">

        {/* 1. DOCUMENT UPLOAD SECTION */}
        <div className="rounded-lg p-6 border border-gray-200 bg-white shadow-sm">
          <h2 className="font-semibold text-lg text-gray-900 mb-4">
            Document Upload
          </h2>

          <form onSubmit={handleUpload} className="space-y-4">
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept=".pdf,.txt,.docx"
                onChange={handleFileChange}
                ref={fileInputRef}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
              />
              <button
                type="submit"
                disabled={!selectedFile || uploadStatus === "Processing..."}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                Upload
              </button>
            </div>

            {/* Visual Status Indicator */}
            <div className="text-sm font-medium">
              Status:{" "}
              {uploadStatus === "Processing..." && (
                <span className="text-amber-600 animate-pulse">Processing...</span>
              )}
              {uploadStatus === "Indexed ✓" && (
                <span className="text-green-600 font-semibold">Indexed ✓</span>
              )}
              {uploadStatus === "Error" && (
                <span className="text-red-600">Failed</span>
              )}
              {uploadStatus === "Ready to upload" && (
                <span className="text-gray-500">Ready to upload</span>
              )}
            </div>
          </form>

          {uploadError && (
            <p className="mt-2 text-sm text-red-600">{uploadError}</p>
          )}

          {/* Document list */}
          {documents.length > 0 && (
            <div className="mt-6 border-t border-gray-100 pt-4">
              <h3 className="font-medium text-sm text-gray-700 mb-2">
                Indexed Documents ({documents.length})
              </h3>
              <ul className="space-y-1.5 text-sm text-gray-600">
                {documents.map((doc, i) => (
                  <li key={doc.id || i} className="flex justify-between items-center">
                    <span className="font-medium text-gray-800">{doc.filename}</span>
                    <span className="text-xs text-gray-400">
                      {doc.upload_timestamp ? new Date(doc.upload_timestamp).toLocaleDateString() : ""}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* 2. CHAT INTERFACE SECTION */}
        <div className="rounded-lg border border-gray-200 bg-white shadow-sm flex flex-col h-[500px]">
          <div className="p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg">
            <h2 className="font-semibold text-lg text-center text-gray-800">
              AI Research Assistant
            </h2>
          </div>

          {/* Scrollable message area */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {messages.length === 0 && (
              <p className="text-center text-gray-400 text-sm mt-12">
                Ask a question to start the conversation.
              </p>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"
                  }`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-2 text-sm ${msg.role === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-gray-100 text-gray-800 rounded-bl-none"
                    }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>

                  {/* Prompt required citation format: 📄 filename — Page X */}
                  {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-200 text-xs space-y-1 text-gray-600">
                      <p className="font-semibold text-gray-700">Sources:</p>
                      {msg.citations.map((cit, i) => (
                        <div key={i} className="flex items-center gap-1">
                          <span>📄 {cit.filename || cit.source} — Page {cit.page || 1}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Thinking indicator */}
            {isThinking && (
              <div className="flex items-start">
                <div className="bg-gray-100 text-gray-500 rounded-xl rounded-bl-none px-4 py-2 text-sm animate-pulse">
                  thinking...
                </div>
              </div>
            )}
          </div>

          {/* Chat input form */}
          <form onSubmit={sendMessage} className="p-3 border-t border-gray-100 flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isThinking}
            />
            <button
              type="submit"
              disabled={isThinking || !inputValue.trim()}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              Send
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}