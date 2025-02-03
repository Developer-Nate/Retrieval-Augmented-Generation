import pickle
from sentence_transformers import SentenceTransformer


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


def save_embeddings_to_file(embeddings_dict, filename):
    """Save the embeddings dictionary to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(embeddings_dict, file)
    print(f"Embeddings saved to {filename}")


def load_embeddings_from_file(filename):
    """Load the embeddings dictionary from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


# Example usage
file_path = "Flatland.txt"  # Path to your text file
chunks = split_into_chunks(file_path)  # Split the file content into chunks
embeddings_dict = generate_embeddings(chunks)  # Generate embeddings for the chunks

# Save the embeddings to a file
output_file = "embeddings.pkl"  # File name where embeddings will be saved
save_embeddings_to_file(embeddings_dict, output_file)

# If you need to load the embeddings from the saved file:
# loaded_embeddings = load_embeddings_from_file(output_file)
# print(loaded_embeddings.get(0))
