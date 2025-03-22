# the following architecture is inspired by https://haystack.deepset.ai/cookbook/query-expansion

import os
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
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
# from haystack.components.rankers import TransformersSimilarityRanker
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from haystack.components.generators import HuggingFaceLocalGenerator
from optimum.quanto import QuantizedModelForCausalLM
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
model = AutoModel.from_pretrained("google/bigbird-pegasus-large-arxiv")

load_dotenv()

# llm = HuggingFaceLocalGenerator(model="thesven/Mistral-7B-Instruct-v0.3-GPTQ", huggingface_pipeline_kwargs={"device_map": "balanced"}, token=Secret.from_env_var("HF_TOKEN"))
llm = GoogleAIGeminiGenerator(model="gemini-2.0-flash-lite", api_key=Secret.from_env_var("GOOGLE_AI_STUDIO"))

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
    documents = [Document(id=document["_id"], content=document["title"] + " " + document["text"], meta={"title": document["title"], **document["metadata"]}) for document in documents]

    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True,
        unicode_normalization="NFKC",
        keep_id=True
    )

    documents = cleaner.run(documents)["documents"]

    splitter = DocumentSplitter(split_by="sentence", split_length=3, split_overlap=0)
    splitter.warm_up()

    for i in range(len(documents)):
        print(f"Embedding document {i + 1}/{len(documents)}...")
        documents[i].embedding = get_vector_embedding(documents[i].content)

        # chunks = splitter.run([documents[i]])["documents"]

        # embeddings = []

        # for j in range(len(chunks)):
        #     embeddings.append(get_vector_embedding(chunks[j].content))

        # # # Perform mean pooling to capture features across multiple chunks
        # # embedding = np.mean(embeddings, axis=0)

        # # Perform max pooling to capture features across multiple chunks
        # embedding = np.max(embeddings, axis=0)

        # documents[i].embedding = embedding.tolist()
        documents[i].content = format_for_bm25(documents[i].content)

        print(documents[i])

    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus", embedding_similarity_function="cosine")

    # Write the embedded documents to the document store
    writer = DocumentWriter(document_store=document_store)
    writer.run(documents)

    document_store.save_to_disk("document_store.json")

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
            "query_embedding": get_vector_embedding(queries[i]["text"]),
            "top_k": 100
        }
    })

    results = results["bert_ranker"]["documents"]

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

    scores.to_csv(r"results_hybrid_bigbird.txt", header=False, index=False, sep=" ")