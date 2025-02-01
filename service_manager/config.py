import configparser

def get_config():
    config = configparser.ConfigParser()
    config.read("service_manager/config.ini")

    milvus = MilvusConfig(
        host=config.get("milvus", "host"),
        port=int(config.get("milvus", "port")),
        collection_name=config.get("milvus", "collection_name"),
        dim=int(config.get("milvus", "dim"))
    )

    llm_model = config.get("llama_index", "model")
    embedder_model = config.get("sentence_transformers", "model")

    return milvus, llm_model, embedder_model
