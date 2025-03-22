import json
from typing import List, Optional, Dict
from preprocessing import format_for_bm25
from haystack import Pipeline, Document, component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    

# this script comes from https://haystack.deepset.ai/cookbook/query-expansion and was modified to work with HuggingFace
@component
class QueryExpander:

    def __init__(self, llm:GoogleAIGeminiGenerator, prompt: Optional[str] = None):
        self.query_expansion_prompt = prompt
        if prompt == None:
          self.query_expansion_prompt = """
          You are part of an information system that processes users queries.
          You expand a given query into {{number}} queries that are similar in meaning using as many different synonyms as possible.
          ONLY return a Python list as a string. Do not elaborate on your answer and do not wrap your answer as Python code.
          For each expanded query, please wrap the string in double quotes (") and NOT single quotes.
          
          Structure:
          Follow the structure shown below in examples to generate expanded queries.
          Examples:
          Example Query 1: "climate change effects"
          Example Expanded Queries: ["impact of climate change", "consequences of global warming", "effects of environmental changes"]
          
          Example Query 2: ""machine learning algorithms""
          Example Expanded Queries: ["neural networks", "clustering", "supervised learning", "deep learning"]
          
          Your Task:
          Query: "{{query}}"
          Example Expanded Queries:
          """
        builder = PromptBuilder(self.query_expansion_prompt)
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=List[str])
    def run(self, query: str, number: int = 5):
        result = self.pipeline.run({'builder': {'query': query, 'number': number}})
        expanded_query = json.loads(result['llm']['replies'][0].strip()) + [query]
        return {"queries": list(expanded_query)}
    
# this script comes from https://haystack.deepset.ai/cookbook/query-expansion
@component
class MultiQueryInMemoryBM25Retriever:

    def __init__(self, retriever: InMemoryBM25Retriever, top_k: int = 100):

        self.retriever = retriever
        # self.results = []
        self.results = {}
        # self.ids = set()
        self.top_k = top_k

    def add_document(self, document: Document):
        """
        Only adds a new document if the document was not already retrieved.
        """
        # if document.id not in self.ids:
        #     self.results.append(document)
        #     self.ids.add(document.id)

        if document.id not in self.results.keys():
            self.results[document.id] = document

    @component.output_types(documents=Dict[str, Document])
    def run(self, queries: List[str], top_k: int = 100):
        if top_k != None:
          self.top_k = top_k

        for query in queries:
          query = format_for_bm25(query)

          result = self.retriever.run(query = query, top_k = self.top_k)
          for doc in result['documents']:
            self.add_document(doc)

        # self.results.sort(key=lambda x: x.score, reverse=True)
        return {"documents": self.results}

# this script was inspired by the scripts from https://haystack.deepset.ai/cookbook/query-expansion
@component
class InMemoryEmbeddingRanker:
    def __init__(self, top_k: int = 100):
        self.results = []
        self.ids = set()
        self.top_k = top_k

    def add_document(self, document: Document):
        """
        Only adds a new document if the document was not already retrieved.
        """
        if document.id not in self.ids:
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding, documents: Dict[str, Document], top_k: int = 100):
        if top_k != None:
          self.top_k = top_k

        document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus", embedding_similarity_function="cosine")
        document_store.write_documents(list(documents.values()))

        self.ranker = InMemoryEmbeddingRetriever(document_store=document_store, scale_score=True, top_k=len(documents.keys()))

        result = self.ranker.run(query_embedding = query_embedding, top_k = self.top_k)

        for doc in result['documents']:
            doc.score = (documents[doc.id].score + doc.score) / 2
            # doc.score = (0.7 * documents[doc.id].score) + (0.3 * doc.score)
            self.add_document(doc)

        self.results.sort(key=lambda x: x.score, reverse=True)
        self.results = self.results[:self.top_k]
        return {"documents": self.results}