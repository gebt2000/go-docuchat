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
	fmt.Println("🚀 Server running on port " + port)
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

	// 1. EMBEDDING STEP
	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{body.Question},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		fmt.Printf("❌ OpenAI Embedding Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("OpenAI Embedding Error: %v", err)})
		return
	}
	questionVector := resp.Data[0].Embedding

	// 2. QDRANT SEARCH STEP
	searchResult, err := qdrantClient.Search(context.Background(), &pb.SearchPoints{
		CollectionName: collectionName,
		Vector:         questionVector,
		Limit:          1,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		fmt.Printf("❌ Qdrant Search Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Qdrant Search Error: %v", err)})
		return
	}

	if len(searchResult.Result) == 0 {
		c.JSON(http.StatusOK, gin.H{"answer": "I couldn't find any relevant info in the document."})
		return
	}

	// SAFETY CHECK: Handle missing payload
	payloadItem, ok := searchResult.Result[0].Payload["text"]
	if !ok || payloadItem == nil {
		fmt.Println("❌ Payload 'text' is missing or nil")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Data missing in database"})
		return
	}
	foundText := payloadItem.GetStringValue()
	
	// 3. CHAT COMPLETION STEP
	prompt := fmt.Sprintf("Context: %s\n\nQuestion: %s\n\nAnswer based ONLY on the context.", foundText, body.Question)
	
	chatResp, err := aiClient.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		fmt.Printf("❌ OpenAI Chat Error: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("OpenAI Chat Error: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"answer":  chatResp.Choices[0].Message.Content,
		"context": foundText,
	})
}

func handleIngest(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	tempPath := filepath.Join(".", file.Filename)
	c.SaveUploadedFile(file, tempPath)
	defer os.Remove(tempPath)

	content, err := readPdf(tempPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "PDF Read Error"})
		return
	}

	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{content},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Embedding Error: %v", err)})
		return
	}
	
	qdrantClient.CreateCollection(context.Background(), &pb.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: &pb.VectorsConfig{Config: &pb.VectorsConfig_Params{Params: &pb.VectorParams{
			Size: 1536,
			Distance: pb.Distance_Cosine,
		}}},
	})

	upsertReq := &pb.UpsertPoints{
		CollectionName: collectionName,
		Points: []*pb.PointStruct{
			{
				Id: &pb.PointId{PointIdOptions: &pb.PointId_Uuid{Uuid: uuid.New().String()}},
				Vectors: &pb.Vectors{VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: resp.Data[0].Embedding}}},
				Payload: map[string]*pb.Value{"text": {Kind: &pb.Value_StringValue{StringValue: content}}},
			},
		},
	}
	_, err = qdrantClient.Upsert(context.Background(), upsertReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Qdrant Upsert Error: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "success"})
}

func setupInfrastructure() {
	godotenv.Load() 
	aiClient = openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	qdrantURL := os.Getenv("QDRANT_URL")
	qdrantKey := os.Getenv("QDRANT_API_KEY")
	
	if qdrantURL == "" { qdrantURL = "localhost:6334" }

	var conn *grpc.ClientConn
	var err error

	if qdrantKey == "" {
		conn, err = grpc.NewClient(qdrantURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		conn, err = grpc.NewClient(qdrantURL, 
			grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})),
			grpc.WithPerRPCCredentials(tokenAuth{token: qdrantKey}),
		)
	}

	if err != nil { log.Fatalf("Qdrant Connect Error: %v", err) }
	qdrantClient = pb.NewPointsClient(conn)
}

type tokenAuth struct { token string }
func (t tokenAuth) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{"api-key": t.token}, nil
}
func (t tokenAuth) RequireTransportSecurity() bool { return true }

func readPdf(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil { return "", err }
	defer f.Close()
	var totalText string
	for pageIndex := 1; pageIndex <= r.NumPage(); pageIndex++ {
		p := r.Page(pageIndex)
		if p.V.IsNull() { continue }
		text, _ := p.GetPlainText(nil)
		totalText += text
	}
	return totalText, nil
}
