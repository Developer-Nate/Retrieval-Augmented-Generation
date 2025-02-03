from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the saved embeddings from a file
def load_embeddings_from_file(filename):
    """Load the embeddings dictionary from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Generate the query embedding using the SentenceTransformer model
def generate_query_embedding(query, model):
    """Generate the embedding for a query."""
    return model.encode(query)

# Find the top N most similar chunks to the query
def find_most_similar_chunks(query, embeddings_dict, model, top_n=3):
    """Find the top N most similar text chunks to the query."""
    query_embedding = generate_query_embedding(query, model)
    
    # Extract the embeddings and corresponding texts from the dictionary
    chunk_embeddings = [entry['embedding'] for entry in embeddings_dict.values()]
    chunk_texts = [entry['text'] for entry in embeddings_dict.values()]
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Get the indices of the top N most similar chunks
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Get the top N chunks and their similarity scores
    top_chunks = [(chunk_texts[i], similarities[i]) for i in top_indices]
    
    return top_chunks

# Generate a response using the FLAN-T5 model
def generate_response(prompt, model, tokenizer):
    """Generate a response from the FLAN-T5 model based on the prompt."""
    # Encode the input prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function to get query input and generate the response
def main():
    # Load the saved embeddings
    embeddings_file = "embeddings.pkl"  # Path to the saved embeddings file
    embeddings_dict = load_embeddings_from_file(embeddings_file)
    
    # Load SentenceTransformer model for embedding generation
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the FLAN-T5 model and tokenizer from HuggingFace
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
    
    # Take query input from the user
    query = input("Enter your query: ")
    
    # Find the top 3 most similar chunks
    top_chunks = find_most_similar_chunks(query, embeddings_dict, sentence_model, top_n=3)
    
    # Combine the top chunks into a single prompt
    prompt = "Answer the following question based on the information provided:\n"
    for i, (text, similarity) in enumerate(top_chunks):
        prompt += f"\nChunk {i+1} (Similarity: {similarity:.4f}): {text}"
    
    prompt += f"\n\nQuestion: {query}\nAnswer:"

    # Generate the response using the FLAN-T5 model
    response = generate_response(prompt, model, tokenizer)
    
    # Print the response
    print("\nGenerated Response:")
    print(response)

# Run the main function
if __name__ == "__main__":
    main()
