import os
from utils import load_jsonl
import numpy as np
import pandas as pd
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
# from haystack.components.rankers import TransformersSimilarityRanker
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

def get_vector_embedding(text:str) -> list:
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
    documents = [Document(id=document["_id"], content=document["title"] + " " + document["text"], meta=document["metadata"]) for document in documents]

    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True,
        unicode_normalization="NFKC"
    )
    cleaner.warm_up()
    documents = cleaner.run(documents)["documents"]

    splitter = DocumentSplitter(split_by="sentence", split_length=3, split_overlap=0)
    splitter.warm_up()

    for i in range(len(documents)):
        print(f"Embedding document {i + 1}/{len(documents)}...")
        chunks = splitter.run([documents[i]])["documents"]

        embeddings = []

        for j in range(len(chunks)):
            embeddings.append(get_vector_embedding(chunks[j].content))

        # Perform mean pooling to capture features across multiple chunks
        embedding = np.mean(embeddings, axis=0)

        documents[i].embedding = embedding.tolist()

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # Write the embedded documents to the document store
    writer = DocumentWriter(document_store=document_store)
    writer.run(documents)

    document_store.save_to_disk("document_store.json")

# pipeline = Pipeline()
# pipeline.add_component("bert_retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=100))

bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=100)

queries = load_jsonl("queries_for_test.jsonl")
scores = pd.DataFrame()

for i in range(len(queries)):
    print(f"Generating results for query {i + 1}/{len(queries)}")

    results = bm25_retriever.run(query=queries[i]["text"])["documents"]

    temp = InMemoryDocumentStore()
    temp.write_documents(results)

    bert_ranker = InMemoryEmbeddingRetriever(document_store=temp, top_k=100)
    results = bert_ranker.run(query_embedding=get_vector_embedding(queries[i]["text"]))["documents"]

    # result = pipeline.run({
    #     "bert_retriever": {
    #         "query_embedding": get_vector_embedding(queries[i]["text"])
    #     }
    # })

    # results = result["bert_retriever"]["documents"]

    for j in range(len(results)):
        row = {
            "ID": queries[i]["_id"],
            "Constant": "Q0",
            "DocID": results[j].id,
            "Rank": j,
            "Score": "{:.6f}".format(results[j].score),
            "RunTag": "run1"
        }

        scores = pd.concat([scores, pd.DataFrame(data=[row])])

scores.to_csv(r"results_hybrid.txt", header=False, index=False, sep=" ")