from embedding_generator import generate_word_embeddings

dataset = "stanfordnlp/sst2"
model_name = "bert-base-uncased"
model_path = ""
batch_size = 1
chunk_size = 1000
output_dir = "./output"
cuda = True
split = "train"
special_embeddings = False
word_embeddings = True


config = {
    "dataset_name": dataset,
    "model_name": model_name,
    "model_path": model_path,
    "batch_size": batch_size,
    "chunk_size": chunk_size,
    "output_dir": output_dir,
    "cuda": cuda,
    "split": split,
    "special_embeddings": special_embeddings,
    "word_embeddings": word_embeddings
}

generate_word_embeddings(config)
