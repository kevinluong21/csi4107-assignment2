# the following architecture is inspired by https://haystack.deepset.ai/cookbook/query-expansion

import time
from utils import load_jsonl, QueryExpander, MultiQueryInMemoryBM25Retriever, InMemoryEmbeddingRanker
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

load_dotenv()

llm = GoogleAIGeminiGenerator(model="gemini-2.0-flash-lite", api_key=Secret.from_env_var("GOOGLE_AI_STUDIO"))

documents = load_jsonl("scifact/corpus.jsonl")

documents = [Document(id=document["_id"], content=document["title"] + " " + document["text"], meta={"title": document["title"], **document["metadata"]}) for document in documents]

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=True,
    unicode_normalization="NFKC",
    keep_id=True
)

text_embedder = SentenceTransformersTextEmbedder()
text_embedder.warm_up()

document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus", embedding_similarity_function="cosine", bm25_parameters={"k": 1.2, "b": 0.5})

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component("cleaner", cleaner)
preprocessing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
preprocessing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

preprocessing_pipeline.connect("cleaner.documents", "embedder.documents")
preprocessing_pipeline.connect("embedder.documents", "writer.documents")

preprocessing_pipeline.run({
    "cleaner": {
        "documents": documents
    }
})

pipeline = Pipeline()
pipeline.add_component("query_expander", QueryExpander(llm=llm))
pipeline.add_component("bm25_retriever", MultiQueryInMemoryBM25Retriever(retriever=InMemoryBM25Retriever(document_store=document_store, scale_score=True), top_k=100))
pipeline.add_component("bert_ranker", InMemoryEmbeddingRanker())

pipeline.connect("query_expander.queries", "bm25_retriever.queries")
pipeline.connect("bm25_retriever.documents", "bert_ranker.documents")

queries = load_jsonl("queries_for_test.jsonl")
scores = pd.DataFrame()

for i in range(len(queries)):
    print(f"Generating results for query {i + 1}/{len(queries)}")

    # To avoid hitting Google AI Studio's rate limits, the program will sleep for a minute every 30 requests
    if (i + 1) % 30 == 0:
        print("Program will sleep for 60 seconds to avoid rate limits...")
        time.sleep(60)

    results = pipeline.run({
        "query_expander": {
            "query": queries[i]["text"],
            "number": 10
        },
        "bm25_retriever": {
            "top_k": 100
        },
        "bert_ranker": {
            "query_embedding": text_embedder.run(queries[i]["text"])["embedding"],
            "top_k": 100
        }
    })

    results = results["bert_ranker"]["documents"]

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

    scores.to_csv(r"results_hybrid_sentence_transformer.txt", header=False, index=False, sep=" ")