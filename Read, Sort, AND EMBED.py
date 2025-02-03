from sentence_transformers import SentenceTransformer


# Function to read and split the file into chunks
def read_and_split_file(file_path):
    """Read the content of a file and split it into chunks by double newlines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()  # Read the entire file content
        chunks = content.split('\n\n')  # Split the content by double newlines
        # Remove any leading/trailing whitespace from each chunk
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks


# Function to generate embeddings for text chunks
def generate_embeddings(text_chunks):
    """Generate embeddings for a list of text chunks using the SentenceTransformer model."""
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the text chunks
    embeddings = model.encode(text_chunks, convert_to_tensor=True)

    # Store embeddings and associated text in a dictionary
    embeddings_dict = {
        "text": text_chunks,
        "embeddings": embeddings
    }
    return embeddings_dict


# Main script
if __name__ == "__main__":
    # Step 1: Read and split the file into chunks
    file_path = 'Flatland.txt'  # Path to the file
    text_chunks = read_and_split_file(file_path)

    # Step 2: Generate embeddings for the text chunks
    embeddings_dict = generate_embeddings(text_chunks)

    # Step 3: Print the embeddings dictionary for demonstration
    for i, (text, embedding) in enumerate(zip(embeddings_dict["text"], embeddings_dict["embeddings"])):
        print(f"Chunk {i + 1}:")
        print(f"Text: {text}")
        print(f"Embedding (first 5 values): {embedding[:5]}")  # Print first 5 values for brevity
        print("-" * 40)