package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/ledongthuc/pdf"
	pb "github.com/qdrant/go-client/qdrant"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// 1. Load API Key
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file. Did you create it?")
	}
	apiKey := os.Getenv("OPENAI_API_KEY")

	// 2. Read PDF
	fmt.Println("📄 Reading PDF...")
	content, err := readPdf("sample.pdf")
	if err != nil {
		log.Fatalf("Error reading PDF: %v (Make sure sample.pdf is in this folder)", err)
	}
	fmt.Printf("   Found %d characters of text.\n", len(content))

	// 3. Connect to OpenAI
	aiClient := openai.NewClient(apiKey)

	// 4. Connect to Qdrant
	conn, err := grpc.NewClient("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect to Qdrant: %v", err)
	}
	defer conn.Close()
	qdrantClient := pb.NewPointsClient(conn)

	// 5. Generate Embedding
	fmt.Println("🧠 Sending text to OpenAI to generate embeddings...")
	resp, err := aiClient.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Input: []string{content},
			Model: openai.SmallEmbedding3,
		},
	)
	if err != nil {
		log.Fatalf("OpenAI Error: %v", err)
	}
	vector := resp.Data[0].Embedding
	fmt.Println("✅ Got Vector from OpenAI!")

	// 6. Store in Qdrant
	collectionName := "pdf_collection"
	upsertReq := &pb.UpsertPoints{
		CollectionName: collectionName,
		Points: []*pb.PointStruct{
			{
				Id: &pb.PointId{
					PointIdOptions: &pb.PointId_Uuid{Uuid: uuid.New().String()},
				},
				Vectors: &pb.Vectors{
					VectorsOptions: &pb.Vectors_Vector{
						Vector: &pb.Vector{Data: vector},
					},
				},
				Payload: map[string]*pb.Value{
					"text": {Kind: &pb.Value_StringValue{StringValue: content}},
				},
			},
		},
	}

	_, err = qdrantClient.Upsert(context.Background(), upsertReq)
	if err != nil {
		log.Fatalf("Qdrant Error: %v", err)
	}

	fmt.Println("🚀 Success! PDF text stored in Vector Database.")
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
