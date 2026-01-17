package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// 1. Connect to Qdrant (The Vector Database)
	// We use port 6334 for gRPC (faster), 6333 is for HTTP
	conn, err := grpc.NewClient("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect: %v", err)
	}
	defer conn.Close()

	// 2. Create the Client
	collections_client := pb.NewCollectionsClient(conn)

	// 3. Define a nice name for our document storage
	collectionName := "pdf_collection"

	// 4. Check if it exists, if not, create it
	// Vector Size: 1536 (Standard for OpenAI's text-embedding-3-small)
	// Distance: Cosine (Standard for text similarity)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	_, err = collections_client.Create(ctx, &pb.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: &pb.VectorsConfig{
			Config: &pb.VectorsConfig_Params{
				Params: &pb.VectorParams{
					Size:     1536,
					Distance: pb.Distance_Cosine,
				},
			},
		},
	})

	if err != nil {
		fmt.Printf("Note: Collection might already exist or error: %v\n", err)
	} else {
		fmt.Println("✅ Success! Created Vector Collection: 'pdf_collection'")
	}
	
	fmt.Println("🚀 Qdrant Database is ready for AI data.")
}
