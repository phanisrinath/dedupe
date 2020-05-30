# Deduplication

This starter deduplication solution is meant to achieve the following:

- Build a reusable framework to perform Deduplication
  - Leverage pre-trained pairwise entity resolution model
  - By providing a trainable object
- Help developers publish this as an endpoint using dockers and Flask


# Architecture of the solution

- Train a pair wise classification problem using custom string edit distance features 
- At inference time leverage Spotify's Annoy for reducing the feature search space and pair wise matches

# Future Work

- Replace hand derived features with a context enhanced features using a language model based embeddings such as BERT or ELMo
- Scalable architecture for faster response and train time
