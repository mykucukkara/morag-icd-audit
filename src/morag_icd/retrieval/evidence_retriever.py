from .hybrid_retriever import HybridRetriever

class ClinicalEvidenceRetriever:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        
    def retrieve_evidence(self, clinical_query: str, top_k: int) -> list[dict]:
        return self.retriever.retrieve(clinical_query, top_k)
