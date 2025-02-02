from dataclasses import dataclass


@dataclass
class MilvusConfig:
    host: str
    port: int
    collection_name: str
    dim: int


ServiceMetadataFields = ["label", "path"]
InvestigationArtifactTypes = ["custom_instruction"]


@dataclass
class VectorDBConfig:
    embeddings: str
