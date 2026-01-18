package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

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
)

func main() {
	// 1. Load API Key
	if err := godotenv.Load(); err != nil {
		log.Fatal("Error loading .env file")
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	aiClient := openai.NewClient(apiKey)

	// 2. Connect to Qdrant
	conn, err := grpc.NewClient("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect to Qdrant: %v", err)
	}
	defer conn.Close()
	qdrantClient := pb.NewPointsClient(conn)

	// 3. DECIDE: Are we teaching (Ingest) or asking (Chat)?
	args := os.Args[1:]
	if len(args) > 0 {
		// User provided a question -> "Chat Mode"
		question := strings.Join(args, " ")
		runChat(aiClient, qdrantClient, question)
	} else {
		// No arguments -> "Ingest Mode"
		runIngest(aiClient, qdrantClient)
	}
}

// --- MODE 1: CHAT ---
func runChat(aiClient *openai.Client, qdrantClient pb.PointsClient, question string) {
	fmt.Printf("❓ Question: %s\n", question)

	// 1. Convert Question to Vector
	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{question},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		log.Fatalf("OpenAI Error: %v", err)
	}
	questionVector := resp.Data[0].Embedding

	// 2. Search Qdrant for the "Nearest" text chunk
	searchResult, err := qdrantClient.Search(context.Background(), &pb.SearchPoints{
		CollectionName: collectionName,
		Vector:         questionVector,
		Limit:          1, // We only want the BEST match
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		log.Fatalf("Search Error: %v", err)
	}

	if len(searchResult.Result) == 0 {
		fmt.Println("❌ No relevant information found in the PDF.")
		return
	}

	// 3. Extract the text found in the database
	foundText := searchResult.Result[0].Payload["text"].GetStringValue()
	score := searchResult.Result[0].Score
	fmt.Printf("🔎 Found context (Similarity: %.2f)\n", score)

	// 4. Send to OpenAI to generate the final answer
	prompt := fmt.Sprintf("Context: %s\n\nQuestion: %s\n\nAnswer the question based ONLY on the context above.", foundText, question)

	chatResp, err := aiClient.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		log.Fatalf("Chat Error: %v", err)
	}

	fmt.Println("\n🤖 AI Answer:")
	fmt.Println(chatResp.Choices[0].Message.Content)
}

// --- MODE 2: INGEST ---
func runIngest(aiClient *openai.Client, qdrantClient pb.PointsClient) {
	fmt.Println("📄 Reading PDF...")
	content, err := readPdf("sample.pdf")
	if err != nil {
		log.Fatalf("Error: %v (Is sample.pdf here?)", err)
	}
	fmt.Printf("   Found %d characters.\n", len(content))

	fmt.Println("🧠 Generating Vector...")
	resp, err := aiClient.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Input: []string{content},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		log.Fatalf("OpenAI Error: %v", err)
	}
	vector := resp.Data[0].Embedding

	fmt.Println("💾 Saving to Qdrant...")
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
	_, err = qdrantClient.Upsert(context.Background(), upsertReq)
	if err != nil {
		log.Fatalf("Qdrant Error: %v", err)
	}
	fmt.Println("🚀 Success! PDF Learned.")
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
