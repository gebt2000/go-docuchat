import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  // 1. Handle File Upload
  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("Uploading...");
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post("http://localhost:8080/ingest", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus("✅ Upload Success! You can now chat.");
    } catch (error) {
      console.error(error);
      setUploadStatus("❌ Upload Failed. Is the Go server running?");
    }
  };

  // 2. Handle Chat Message
  const handleChat = async () => {
    if (!question) return;

    // Add user message to UI immediately
    const userMessage = { role: "user", content: question };
    setChatHistory((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8080/chat", {
        question: userMessage.content,
      });

      // Add AI response to UI
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
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "20px", fontFamily: "Arial" }}>
      <h1>🤖 Go-DocuChat</h1>
      
      {/* Upload Section */}
      <div style={{ marginBottom: "20px", padding: "15px", border: "1px solid #ccc", borderRadius: "8px" }}>
        <h3>1. Upload Knowledge</h3>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} style={{ marginLeft: "10px" }}>Upload PDF</button>
        <p style={{ marginTop: "10px", fontWeight: "bold" }}>{uploadStatus}</p>
      </div>

      {/* Chat Section */}
      <div style={{ border: "1px solid #ccc", borderRadius: "8px", padding: "15px", height: "400px", overflowY: "scroll", background: "#f9f9f9" }}>
        {chatHistory.length === 0 && <p style={{ color: "#888", textAlign: "center" }}>Ask a question about your PDF...</p>}
        
        {chatHistory.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.role === "user" ? "right" : "left", marginBottom: "10px" }}>
            <div style={{ 
              display: "inline-block", 
              padding: "10px", 
              borderRadius: "10px", 
              background: msg.role === "user" ? "#007bff" : "#e0e0e0", 
              color: msg.role === "user" ? "#fff" : "#000",
              maxWidth: "80%"
            }}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <p>Thinking...</p>}
      </div>

      {/* Input Section */}
      <div style={{ marginTop: "20px", display: "flex" }}>
        <input 
          type="text" 
          value={question} 
          onChange={(e) => setQuestion(e.target.value)} 
          onKeyPress={(e) => e.key === 'Enter' && handleChat()}
          placeholder="Type your question..." 
          style={{ flex: 1, padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} 
        />
        <button onClick={handleChat} style={{ marginLeft: "10px", padding: "10px 20px" }}>Send</button>
      </div>
    </div>
  );
}

export default App;
