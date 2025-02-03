# Retrieval-Augmented-Generation

Selected Document: 
Flatland: A romance of many dimensions

Flatland.txt: A reprint of the fictional work Flatland: a romance of Many Directions


Program explanation :
  
  The program works by using sentence embeddings to represent sentences or pieces of text as high-dimensional vectors. These embeddings are generated using a model like sentence-transformers, which is a         type of neural network trained to capture the semantic meaning of sentences. Once each sentence is transformed into an embedding, the program can compare these embeddings using cosine similarity, a metric      that measures how similar two vectors are by calculating the cosine of the angle between them.
    If the cosine similarity score between two sentence embeddings is high, it means the sentences are semantically similar, and the program can use this information for tasks like semantic search, text           comparison, or clustering. In essence, the program turns sentences into numbers (embeddings) and then compares these numbers to understand the relationships between different pieces of text.

Five important Questions: 
To fully understand how a program using cosine similarity, sentence-transformers, and embeddings works, here are five important questions you could ask:
  1. What is the role of embeddings in this program?

       •	Embeddings are numerical representations of words, sentences, or paragraphs that capture semantic meaning. Understanding how they are generated, used, and stored will give you insight into the core             function of the program.
  3. How does sentence-transformers generate embeddings?

       •	Sentence-transformers are pre-trained models that convert sentences into vector embeddings. Knowing the architecture behind the sentence-transformers (e.g., BERT-based models, cross-encoders, etc.)             will help you understand the quality and limitations of the embeddings they produce.
  5. What is cosine similarity, and why is it used in this program?
   
     •	Cosine similarity measures how similar two vectors are by calculating the cosine of the angle between them. It's essential for understanding how the program compares different sentence embeddings.             Understanding why cosine similarity is chosen over other distance metrics (like Euclidean distance) will help clarify how the program determines similarity.
  6. How does the program handle out-of-vocabulary words or unseen phrases?

      •	Embedding models often have a limited vocabulary or ability to generalize. Knowing how the program addresses out-of-vocabulary terms or phrases (such as using subword tokenization or other techniques)         will help you understand the program’s robustness.
  8. How does the program deal with the potential biases or limitations in the sentence-transformers model?

      •	Models like sentence-transformers can have biases or limitations based on the training data. Asking how the program mitigates these issues (e.g., through fine-tuning, careful selection of pre-trained           models, etc.) will help you understand the quality and fairness of the results it produces.

By exploring these questions, you should gain a solid understanding of how the program uses sentence embeddings, cosine similarity, and transformers to perform tasks like semantic search, text matching, or any other functionality it is designed for.


How the system works: The system can retrieve specific text chunks in the .txt file, but seems to fail when attempting to grasp the nuances in the story. This can be attributed to a number of factors: prompts used to generate the code, individual input when naming code files, and the clarity of the text itself. 

Quality of responses: Responses for literal chunks are high quality, however queries about the nature of the topics in the text are…simplistic. For example, when questioning the nature of space in Flatlands, the answer is the literal title rather than an examinations of the ideas therein. 

Improvements or extensions: The system can be improved by generalizing the prompt, or allowing more thorough examination of the themes in the story rather than examinations of specific chunks of text looking to fit a certain query criteria. 

    Enter your query:  What is the nature of space in Flatland?

    Generated Response:
    A romance of many dimensions

    Enter your query: What is the nature of space in Flatlands?
    
    Generated Response:
    clearer
