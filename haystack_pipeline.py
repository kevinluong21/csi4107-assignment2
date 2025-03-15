import os
from utils import load_jsonl
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
# from haystack.components.rankers import TransformersSimilarityRanker
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

def get_vector_embedding(text:str):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output.last_hidden_state[:, 0, :][0].tolist()

if os.path.exists("document_store.json"):
    print("Loading existing document store...")
    document_store = InMemoryDocumentStore()
    document_store = document_store.load_from_disk("document_store.json")

else:
    print("Creating a new document store...")
    documents = load_jsonl("scifact/corpus.jsonl")

    # For each document in the corpus, create a Document object
    documents = [Document(id=document["_id"], content=document["title"] + ". " + document["text"], meta=document["metadata"]) for document in documents]

    splitter = DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1)
    splitter.warm_up()

    documents = splitter.run(documents)["documents"]

    for i in range(len(documents)):
        print(f"Embedding document {i + 1}/{len(documents)}...")
        documents[i].embedding = get_vector_embedding(documents[i].content)

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # Write the embedded documents to the document store
    writer = DocumentWriter(document_store=document_store)
    writer.run(documents)

    document_store.save_to_disk("document_store.json")

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=100))

query = "0-dimensional biomaterials show inductive properties."
query = get_vector_embedding(query)

result = pipeline.run({
    "retriever": {
        "query_embedding": query
    }
})

results = result["retriever"]["documents"]
results = [document.id for document in results]
print("31715818" in results)