def read_and_split_file(file_path):
    """Read the content of a file and split it into chunks by double newlines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()  # Read the entire file content
        chunks = content.split('\n\n')  # Split the content by double newlines
        # Remove any leading/trailing whitespace from each chunk
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

# Example usage
file_path = 'Flatland.txt'  # Updated file name
chunks = read_and_split_file(file_path)

# Print the chunks for demonstration
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n{'-' * 40}")
