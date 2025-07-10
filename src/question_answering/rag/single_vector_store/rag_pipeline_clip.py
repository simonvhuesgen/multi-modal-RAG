import pandas as pd
import logging
import os
from question_answering.rag.single_vector_store.rag_chain import MultimodalRAGChain
from question_answering.rag.single_vector_store.retrieval import ClipRetriever
from langchain_openai import AzureChatOpenAI
from typing import List
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import IMAGES_DIR, INPUT_DATA, MODEL_TYPE, VECTORSTORE_PATH_CLIP_SINGLE


class MultimodalRAGPipelineClip:
    """
    Initializes the Multimodal RAG pipeline.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Uses CLIP as multimodal embedding model to embed the query, the images, and the texts into a single vector store.
    
    Attributes:
        model_type (str): Type of the model to use for answer synthesis.
        store_path (str): Path to the directory where the vector database is stored.
        model: The model used for answer synthesis loaded based on `model_type`.
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        clip_retriever (ClipRetriever): Retrieval using CLIP embeddings for images.
        rag_chain (MultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path):
        
        config = get_azure_config()
        
        if model_type in config:
            print("Using Azure model")
            azure_llm_config = config[model_type]
            self.model = AzureChatOpenAI(
                openai_api_version=azure_llm_config["openai_api_version"],
                azure_endpoint=azure_llm_config["openai_endpoint"],
                azure_deployment=azure_llm_config["deployment_name"],
                model=azure_llm_config["model_version"],
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                max_tokens=400)
            self.tokenizer = None
            
        else:
            print("Using LLaVA model")
            self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")

        self.clip_retriever = ClipRetriever(vectorstore_dir=store_path)
        self.rag_chain = MultimodalRAGChain(self.model, self.tokenizer, self.clip_retriever.retriever)
    
    
    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # drop duplicates
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        return texts
    
    def summarize_data(self, texts: List[str]) -> List[str]:
        text_summaries = self.text_summarizer.summarize(texts)
        return text_summaries   
    
    def index_data(self, texts_df: pd.DataFrame, images_dir: str):
        self.clip_retriever.add_documents(images_dir=images_dir, texts_df=texts_df)

    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = MultimodalRAGPipelineClip(model_type=MODEL_TYPE, store_path=VECTORSTORE_PATH_CLIP_SINGLE)
    texts_df = pipeline.load_data(INPUT_DATA)
    pipeline.index_data(texts_df=texts_df, images_dir=IMAGES_DIR)

    question = "I want to change the behaviour of the stations to continue, if a moderate error occurrs. How can I do this?"
    answer = pipeline.answer_question(question)
    relevant_docs = pipeline.rag_chain.retrieved_docs  
    print("Retrieved images:", len(relevant_docs["images"]), ", Retrieved texts:", len(relevant_docs["texts"]))  
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
