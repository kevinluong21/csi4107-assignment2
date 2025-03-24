# the following architecture is inspired by https://haystack.deepset.ai/cookbook/query-expansion

import time
from utils import load_jsonl, BM25Formatter, QueryExpander, MultiQueryInMemoryBM25Retriever, BM25AndEmbedderRanker
from preprocessing import format_for_bm25
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from haystack import Document, Pipeline
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.rankers import TransformersSimilarityRanker

# An .env file is required to run the LLM
load_dotenv()

# Load the Gemini LLM from Google AI Studio and connect it to Haystack
llm = GoogleAIGeminiGenerator(model="gemini-2.0-flash-lite", api_key=Secret.from_env_var("GOOGLE_AI_STUDIO"))

documents = load_jsonl("scifact/corpus.jsonl")

# For each document in the corpus, create a Document object and store it in a list
documents = [Document(id=document["_id"], content=document["title"] + " " + document["text"], meta={"title": document["title"], **document["metadata"]}) for document in documents]

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=True,
    unicode_normalization="NFKC",
    keep_id=True
)

# Instantiate a document store with parameters for BM25 and embedding similarity
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus", embedding_similarity_function="cosine", bm25_parameters={"k": 1.2, "b": 0.75})

# Pre-processing pipeline involves cleaning, embedding (which stores it in the document's vector_embedding attribute), formatting the content for BM25, and writing all of these documents into the document store
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component("cleaner", cleaner)
preprocessing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
preprocessing_pipeline.add_component("formatter", BM25Formatter())
preprocessing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

preprocessing_pipeline.connect("cleaner.documents", "embedder.documents")
preprocessing_pipeline.connect("embedder.documents", "formatter.documents")
preprocessing_pipeline.connect("formatter.documents", "writer.documents")

preprocessing_pipeline.run({
    "cleaner": {
        "documents": documents
    }
})

# The BM25 pipeline involves using an LLM to expand the query and then perform BM25 retrieval and rank them
bm25_pipeline = Pipeline()
bm25_pipeline.add_component("query_expander", QueryExpander(llm=llm))
bm25_pipeline.add_component("bm25_retriever", MultiQueryInMemoryBM25Retriever(retriever=InMemoryBM25Retriever(document_store=document_store, scale_score=True)))

bm25_pipeline.connect("query_expander.queries", "bm25_retriever.queries")

# The Embedding pipeline involves embedding the query using teh same model as the document embedder and retrieving all documents that are cosine similar to the document and ranking them
embedding_pipeline = Pipeline()
embedding_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
embedding_pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=document_store, scale_score=True, top_k=100))

embedding_pipeline.connect("text_embedder", "embedding_retriever")

# The ranking pipeline involves doing a weighted sum of documents retrieved by BM25 and by Embeddings to get a final score (by default, we weight the embedding score more than the BM25 score) and then re-ranking them again using a transformers model.
ranking_pipeline = Pipeline()
ranking_pipeline.add_component("bm25_embedder_ranker", BM25AndEmbedderRanker())
ranking_pipeline.add_component("transformers_ranker", TransformersSimilarityRanker())

ranking_pipeline.connect("bm25_embedder_ranker.documents", "transformers_ranker.documents")

queries = load_jsonl("queries_for_test.jsonl")
scores = pd.DataFrame()

# Run all 3 pipelines for each query and then save the results in a txt file
for i in range(len(queries)):
    print(f"Generating results for query {i + 1}/{len(queries)}")

    # To avoid hitting Google AI Studio's rate limits, the program will sleep for a minute every 30 requests
    if (i + 1) % 30 == 0:
        print("Program will sleep for 60 seconds to avoid rate limits...")
        time.sleep(60)

    bm25_docs = bm25_pipeline.run({
        "query_expander": {
            "query": queries[i]["text"],
            "number": 5
        },
        "bm25_retriever": {
            "top_k": 100
        }
    })

    bm25_docs = bm25_docs["bm25_retriever"]["documents"]

    embedding_docs = embedding_pipeline.run({
        "text_embedder": {
            "text": queries[i]["text"]
        }
    })

    embedding_docs = embedding_docs["embedding_retriever"]["documents"]

    results = ranking_pipeline.run({
        "bm25_embedder_ranker": {
            "bm25_docs": bm25_docs,
            "embedding_docs": embedding_docs,
            "top_k": 100
        },
        "transformers_ranker": {
            "query": queries[i]["text"],
            "top_k": 100
        }
    })

    results = results["transformers_ranker"]["documents"]

    for j in range(len(results)):
        row = {
            "ID": queries[i]["_id"],
            "Constant": "Q0",
            "DocID": results[j].id,
            "Rank": j + 1,
            "Score": "{:.6f}".format(results[j].score),
            "RunTag": "run1"
        }

        scores = pd.concat([scores, pd.DataFrame(data=[row])])

    scores.to_csv(r"results_triad.txt", header=False, index=False, sep=" ")