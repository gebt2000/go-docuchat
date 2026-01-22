package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/ledongthuc/pdf"
	pb "github.com/qdrant/go-client/qdrant"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	collectionName = "pdf_collection"
	aiClient       *openai.Client
	qdrantClient   pb.PointsClient
)

func main() {
	setupInfrastructure()

	r := gin.Default()
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	r.Use(cors.New(config))

	r.POST("/ingest", handleIngest)
	r.POST("/chat", handleChat)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Println("🚀 Server running on port " + port)
	r.Run(":" + port)
}

func handleChat(c *gin.Context) {
	var body struct {
		Question string `json:"question"`
	}
	if err := c.BindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
		return
	}

	// 1. EMBEDDING
	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{body.Question},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		fmt.Printf("❌ Embed Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "AI Embedding Failed"})
		return
	}

	// 2. SEARCH
	searchResult, err := qdrantClient.Search(context.Background(), &pb.SearchPoints{
		CollectionName: collectionName,
		Vector:         resp.Data[0].Embedding,
		Limit:          1,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		fmt.Printf("❌ Search Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Database Search Failed"})
		return
	}

	if len(searchResult.Result) == 0 {
		c.JSON(http.StatusOK, gin.H{"answer": "No relevant info found."})
		return
	}
	
	payloadText := ""
	if item, ok := searchResult.Result[0].Payload["text"]; ok {
		payloadText = item.GetStringValue()
	}

	// 3. CHAT - GPT-4o-MINI (The Fix)
	chatResp, err := aiClient.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Context: " + payloadText + "\n\nQuestion: " + body.Question},
		},
	})
	if err != nil {
		fmt.Printf("❌ Chat Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("AI Chat Failed: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"answer": chatResp.Choices[0].Message.Content})
}

func handleIngest(c *gin.Context) {
	file, _ := c.FormFile("file")
	tempPath := filepath.Join(".", file.Filename)
	c.SaveUploadedFile(file, tempPath)
	defer os.Remove(tempPath)
	content, _ := readPdf(tempPath)

	resp, _ := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{content},
		Model: openai.SmallEmbedding3,
	})
	
	// Note: We skip CreateCollection because it already exists from your previous successful run.

	qdrantClient.Upsert(context.Background(), &pb.UpsertPoints{
		CollectionName: collectionName,
		Points: []*pb.PointStruct{{
			Id: &pb.PointId{PointIdOptions: &pb.PointId_Uuid{Uuid: uuid.New().String()}},
			Vectors: &pb.Vectors{VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: resp.Data[0].Embedding}}},
			Payload: map[string]*pb.Value{"text": {Kind: &pb.Value_StringValue{StringValue: content}}},
		}},
	})
	c.JSON(http.StatusOK, gin.H{"status": "success"})
}

func setupInfrastructure() {
	godotenv.Load() 
	aiClient = openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	qdrantURL := os.Getenv("QDRANT_URL")
	if qdrantURL == "" { qdrantURL = "localhost:6334" }
	
	var conn *grpc.ClientConn
	var err error
	if os.Getenv("QDRANT_API_KEY") == "" {
		conn, err = grpc.NewClient(qdrantURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		conn, err = grpc.NewClient(qdrantURL, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})), grpc.WithPerRPCCredentials(tokenAuth{token: os.Getenv("QDRANT_API_KEY")}))
	}
	if err != nil { log.Fatalf("Qdrant Connect Error: %v", err) }
	qdrantClient = pb.NewPointsClient(conn)
}

type tokenAuth struct { token string }
func (t tokenAuth) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) { return map[string]string{"api-key": t.token}, nil }
func (t tokenAuth) RequireTransportSecurity() bool { return true }

func readPdf(path string) (string, error) {
	f, r, _ := pdf.Open(path)
	defer f.Close()
	var totalText string
	for i := 1; i <= r.NumPage(); i++ { text, _ := r.Page(i).GetPlainText(nil); totalText += text }
	return totalText, nil
}
