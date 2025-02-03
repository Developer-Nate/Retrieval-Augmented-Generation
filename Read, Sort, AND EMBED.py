from sentence_transformers import SentenceTransformer
import os


def split_into_chunks(file_path):
    """Read the content of the file and split it into chunks separated by double newlines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    chunks = content.split("\n\n")
    return chunks


def generate_embeddings(chunks):
    """Generate embeddings for a list of text chunks."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the pre-trained model
    embeddings_dict = {}

    # Generate embeddings for each chunk
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)  # Get the embedding for the chunk
        embeddings_dict[i] = {
            'text': chunk,
            'embedding': embedding
        }

    return embeddings_dict


# Example usage
file_path = "Flatland.txt"  # Path to your text file
chunks = split_into_chunks(file_path)  # Split the file content into chunks
embeddings_dict = generate_embeddings(chunks)  # Generate embeddings for the chunks

# Print the embeddings of the first chunk as an example
first_chunk = embeddings_dict.get(0)
if first_chunk:
    print("Text:", first_chunk['text'])
    print("Embedding:", first_chunk['embedding'][:10])  # Print only the first 10 values of the embedding
else:
    print("No content found.")

# Example usage
file_path = "Flatland.txt"  # Path to your text file
chunks = split_into_chunks(file_path)
embeddings_dict = generate_embeddings(chunks)

# Print the embeddings of the first chunk as an example
first_chunk = embeddings_dict.get(0)
if first_chunk:
    print("Text:", first_chunk['text'])
    print("Embedding:", first_chunk['embedding'][:10])  # Print only the first 10 values of the embedding
else:
    print("No content found.")
