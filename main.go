package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/ledongthuc/pdf"
	pb "github.com/qdrant/go-client/qdrant"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	collectionName = "pdf_collection"
	aiClient       *openai.Client
	qdrantClient   pb.PointsClient
)

func main() {
	// 1. Setup Environment & Clients
	setupInfrastructure()

	// 2. Setup Web Server (Gin)
	r := gin.Default()

	// 3. Define the Routes (The API Endpoints)
	r.POST("/ingest", handleIngest)
	r.POST("/chat", handleChat)

	// 4. Start the Server
	fmt.Println("🚀 Server running on http://localhost:8080")
	r.Run(":8080")
}

// --- HANDLER 1: Chat ---
// Accepts JSON: {"question": "..."}
// Returns JSON: {"answer": "..."}
func handleChat(c *gin.Context) {
	var body struct {
		Question string `json:"question"`
	}

	if err := c.BindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
		return
	}

	// 1. Vectorize Question
	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{body.Question},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "OpenAI Error: " + err.Error()})
		return
	}
	questionVector := resp.Data[0].Embedding

	// 2. Search Qdrant
	searchResult, err := qdrantClient.Search(context.Background(), &pb.SearchPoints{
		CollectionName: collectionName,
		Vector:         questionVector,
		Limit:          1,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Qdrant Error: " + err.Error()})
		return
	}

	if len(searchResult.Result) == 0 {
		c.JSON(http.StatusOK, gin.H{"answer": "I don't have enough information to answer that."})
		return
	}

	// 3. Generate Answer
	foundText := searchResult.Result[0].Payload["text"].GetStringValue()
	prompt := fmt.Sprintf("Context: %s\n\nQuestion: %s\n\nAnswer based ONLY on the context.", foundText, body.Question)

	chatResp, err := aiClient.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat Error: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"answer":  chatResp.Choices[0].Message.Content,
		"context": foundText,
	})
}

// --- HANDLER 2: Ingest ---
// Accepts Multipart Form: file="resume.pdf"
func handleIngest(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}

	// Save file temporarily
	tempPath := filepath.Join(".", file.Filename)
	if err := c.SaveUploadedFile(file, tempPath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
		return
	}
	defer os.Remove(tempPath) // Cleanup after we are done

	// Read & Embed
	content, err := readPdf(tempPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read PDF"})
		return
	}

	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{content},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "OpenAI Error"})
		return
	}
	vector := resp.Data[0].Embedding

	// Store
	upsertReq := &pb.UpsertPoints{
		CollectionName: collectionName,
		Points: []*pb.PointStruct{
			{
				Id: &pb.PointId{PointIdOptions: &pb.PointId_Uuid{Uuid: uuid.New().String()}},
				Vectors: &pb.Vectors{VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: vector}}},
				Payload: map[string]*pb.Value{"text": {Kind: &pb.Value_StringValue{StringValue: content}}},
			},
		},
	}
	qdrantClient.Upsert(context.Background(), upsertReq)

	c.JSON(http.StatusOK, gin.H{"status": "success", "message": "File ingested successfully", "chars": len(content)})
}

// --- HELPERS ---
func setupInfrastructure() {
	if err := godotenv.Load(); err != nil {
		log.Fatal("Error loading .env file")
	}
	aiClient = openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	conn, err := grpc.NewClient("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect to Qdrant: %v", err)
	}
	qdrantClient = pb.NewPointsClient(conn)
}

func readPdf(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var totalText string
	for pageIndex := 1; pageIndex <= r.NumPage(); pageIndex++ {
		p := r.Page(pageIndex)
		if p.V.IsNull() {
			continue
		}
		text, _ := p.GetPlainText(nil)
		totalText += text
	}
	return totalText, nil
}
