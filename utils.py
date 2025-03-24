import json
import pandas as pd
from typing import List, Optional, Dict
from preprocessing import format_for_bm25
from haystack import Pipeline, Document, component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

def load_jsonl(file_path) -> list[dict]:
    """
    Load a JSONL file in as a list of dictionaries.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
@component
class BM25Formatter:
    """
    This component is responsible for reformatting every document in a list of documents, given a list of documents.
    """
    def __init__(self):
        pass

    @component.output_types(documents=List[Document])
    def run(self, documents:List[Document]):
        for i in range(len(documents)):
            documents[i].content = format_for_bm25(documents[i].content)

        return {"documents": documents}
        

# this script comes from https://haystack.deepset.ai/cookbook/query-expansion and was modified to work with HuggingFace
@component
class QueryExpander:
    """
    This component uses an LLM to expand the given query using synonyms.
    """

    def __init__(self, llm:GoogleAIGeminiGenerator, prompt: Optional[str] = None):
        self.query_expansion_prompt = prompt
        if prompt == None:
          self.query_expansion_prompt = """
          You are part of an information system that processes users queries.
          You expand a given query into {{number}} queries that are similar in meaning as a Python list. Please use as MANY synonyms from biomedical, clinical, physical, and scientific fields as possible.
          You MUST return a Python list as a string!
          Do not elaborate your answer.
          Do not wrap your answer as Python code.
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

        error = True

        # If the LLM accidentally returns a response that is NOT a Python list, loop back and run the pipeline again
        while error:
            try:
                expanded_query = json.loads(result['llm']['replies'][0].strip()) + [query]
                print(list(expanded_query))
                error = False
                return {"queries": list(expanded_query)}
            except:
                pass
    
# this script comes from https://haystack.deepset.ai/cookbook/query-expansion
@component
class MultiQueryInMemoryBM25Retriever:
    """
    This component uses BM25 to retrieve documents for every query in the list of expanded queries and the original query. The resulting list of documents is a union of all documents that were retrieved across all queries.
    """

    def __init__(self, retriever: InMemoryBM25Retriever):
        self.retriever = retriever
        self.results = {}

    def add_document(self, document: Document):
        # Check if the document is already stored in the results. If not, add the new document. If it is, then take the max of either the new score or the score stored in the results.
        if document.id not in self.results.keys():
            self.results[document.id] = document
        else:
            self.results[document.id].score = max(self.results[document.id].score, document.score)

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = 100):
        if top_k != None:
          self.top_k = top_k

        for query in queries:
          query = format_for_bm25(query)

          result = self.retriever.run(query = query, top_k = self.top_k)
          for doc in result['documents']:
            self.add_document(doc)

        documents = list(self.results.values())

        documents.sort(key=lambda x: x.score, reverse=True)

        self.results = {}
        return {"documents": documents}
    
@component
class BM25AndEmbedderRanker:
    """
    This component performs a weighted sum of the BM25 scores and the Embedding scores and then ranks all of the documents based on the new score.
    """
    def __init__(self):
        pass
    
    @component.output_types(documents=List[Document])
    def run(self, bm25_docs: List[Document], embedding_docs: List[Document], bm25_weight:float = 0.3, embedding_weight:float = 0.7, top_k: int=100):
        bm25_scores = [{"ID": document.id, "BM25Score": document.score} for document in bm25_docs]
        embedding_scores = [{"ID": document.id, "EmbeddingScore": document.score} for document in embedding_docs]

        bm25_docs = {document.id: document for document in bm25_docs}
        documents = bm25_docs | {document.id: document for document in embedding_docs if document.id not in bm25_docs.keys()}

        bm25_scores = pd.DataFrame(data=bm25_scores)
        embedding_scores = pd.DataFrame(data=embedding_scores)

        # Convert the list of documents into a pandas dataframe to merge on the IDs and perform a weighted sum, sorting, and then taking the top_k documents.
        results = pd.merge(left=bm25_scores, right=embedding_scores, left_on="ID", right_on="ID", how="outer")
        results = results.fillna(value=0)
        results["WeightedScore"] = (bm25_weight * results["BM25Score"]) + (embedding_weight * results["EmbeddingScore"])
        results = results.sort_values(by="WeightedScore", ascending=False).reset_index(drop=True)
        results = results.iloc[:top_k]

        # Using the IDs of only the top_k documents, return a list of documents (they do not need to be ranked because a transformers ranker will re-rank them again).
        document_ids = set(results["ID"].to_list())
        documents = [document for id, document in documents.items() if id in document_ids]

        return {"documents": documents}

