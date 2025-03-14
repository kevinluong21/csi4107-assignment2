import os
from utils import load_jsonl
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

def get_vector_embedding(text:str):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output.last_hidden_state[:, 0, :][0].tolist()

if os.path.exists("document_store.json"):
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    document_store.load_from_disk("document_store.json")

else:
    documents = load_jsonl("scifact/corpus.jsonl")

    # For each document in the corpus, create a Document object and use BERT to embed the content
    documents = [Document(id=document["_id"], content=document["title"] + " " + document["text"], embedding=get_vector_embedding(document["title"] + " " + document["text"]), meta=document["metadata"]) for document in documents]

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # Write the embedded documents to the document store
    writer = DocumentWriter(document_store=document_store)
    writer.run(documents)

    document_store.save_to_disk("document_store.json")

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryEmbeddingRetriever())
pipeline.add_component("ranker", TransformersSimilarityRanker())
pipeline.connect("retriever", "ranker")