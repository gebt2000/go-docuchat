import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

// AUTOMATIC SWITCH: Uses Cloud URL if set, otherwise falls back to localhost
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("Uploading...");
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post(`${API_URL}/ingest`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus("✅ Ready to chat!");
    } catch (error) {
      console.error(error);
      setUploadStatus("❌ Upload Failed");
    }
  };

  const handleChat = async () => {
    if (!question.trim()) return;

    const userMessage = { role: "user", content: question };
    setChatHistory((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chat`, {
        question: userMessage.content,
      });

      const aiMessage = { role: "ai", content: res.data.answer };
      setChatHistory((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage = { role: "ai", content: "⚠️ Error connecting to server." };
      setChatHistory((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>🤖 DocuChat V2</h1>
      </header>
      
      <div className="upload-section">
        <input type="file" className="file-input" onChange={(e) => setFile(e.target.files[0])} />
        <button className="upload-btn" onClick={handleUpload}>Upload PDF</button>
        {uploadStatus && <span className="status-text">{uploadStatus}</span>}
      </div>

      <div className="chat-window">
        {chatHistory.length === 0 ? (
          <div className="empty-state">
            <p>Upload a document to get started.</p>
          </div>
        ) : (
          chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="bubble">{msg.content}</div>
            </div>
          ))
        )}
        {loading && <div className="message ai"><div className="bubble">Thinking...</div></div>}
        <div ref={chatEndRef} />
      </div>

      <div className="input-area">
        <input 
          type="text" 
          className="chat-input"
          value={question} 
          onChange={(e) => setQuestion(e.target.value)} 
          onKeyPress={(e) => e.key === 'Enter' && handleChat()}
          placeholder="Ask a question..." 
          disabled={loading}
        />
        <button className="send-btn" onClick={handleChat} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

export default App;
