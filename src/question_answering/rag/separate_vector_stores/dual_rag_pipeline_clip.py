import logging
import os
import gc
import pandas as pd
from typing import List
from data_summarization.context_summarization import TextSummarizer
from langchain_openai import AzureChatOpenAI
from question_answering.rag.separate_vector_stores.dual_rag_chain import DualMultimodalRAGChain
from question_answering.rag.separate_vector_stores.dual_retrieval import DualClipRetriever
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import EMBEDDING_MODEL_TYPE, IMAGES_DIR, INPUT_DATA, MODEL_TYPE, TEXT_SUMMARIES_CACHE_DIR, VECTORSTORE_PATH_CLIP_SEPARATE


class DualMultimodalRAGPipelineClip:
    """
    Initializes the Multimodal RAG pipeline with separate vector stores and retrievers for texts and images.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Uses CLIP as multimodal embedding model to embed the images.
    Uses a text embedding model to embed the text.
    The query is embedded both with CLIP and with the text embedding model to allow retrieval for each modality.
    
    Attributes:  
        model_type (str): Type of the model to use for answer synthesis.  
        store_path (str): Path to the directory where the vector databases for texts and images are stored.  
        model: The model used for answer synthesis loaded based on `model_type`.  
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        dual_retriever (DualClipRetriever): Retrieval using CLIP embeddings for images and text embeddings for texts.
        rag_chain (DualMultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path, text_embedding_model):
        
        config = get_azure_config()
        
        if model_type in config:
            print("Using Azure model for answer generation")
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
            print("Using LLaVA model for answer generation")
            self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")

        #self.text_summarizer = TextSummarizer(model_type="gpt4", cache_path=TEXT_SUMMARIES_CACHE_DIR)
        self.dual_retriever = DualClipRetriever(store_path=store_path,
                                            text_model_id=model_type,
                                            text_embedding_model=text_embedding_model)

        self.rag_chain = DualMultimodalRAGChain(self.model,
                                            self.tokenizer,
                                            self.dual_retriever.text_retriever,
                                            self.dual_retriever.img_retriever)

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        print("dataset looks like this:",df.shape, df.columns)
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        return texts


    def load_new_df(self, input_df: pd.DataFrame):
        texts = input_df[input_df["type"] == "text"]
        images = input_df[input_df["type"] == "image"]
        print(input_df.columns)
        text_records = []
        for _, row in texts.iterrows():
            text_records.append({
                "text": row["text_content"],
                "doc_id": row["doc_id"],
                "unique_id": str(row["doc_id"]) + "_" + str(row["chunk_index"])
            })


        

        batch_size = 100  
        processed_count = 0
        
        
        image_records = []
        for _, row in images.iterrows():
            path = row["original_image_path"]
            if path[:5] == "s3://":
                path = "data/399k_imgs/" + str(path.split("/")[-1])        
            with open("../../" + path, "rb") as f:
                image_bytes = f.read()
            image_records.append({
                "image_bytes": image_bytes,
                "doc_id": row["doc_id"],
                "unique_id": path,
                "image_summary": row["image_summary"]
            })

            processed_count += 1
            if processed_count % batch_size == 0:
                gc.collect()
                #print(f"Processed {processed_count} images")
        
        return pd.DataFrame(text_records), pd.DataFrame(image_records)

        

    def summarize_data(self, texts: List[str]) -> List[str]:
        text_summaries = self.text_summarizer.summarize(texts)
        return text_summaries
    
    def index_data(self, images_dir: str, texts: List[str], text_filenames: List[str], text_summaries: List[str]=None):
        self.dual_retriever.add_images(images_dir)   
        if text_summaries:
            self.dual_retriever.add_texts(text_summaries, texts, text_filenames)
        else:
            self.dual_retriever.add_texts(texts, texts, text_filenames)

    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = DualMultimodalRAGPipelineClip(model_type=MODEL_TYPE,
                                     store_path=VECTORSTORE_PATH_CLIP_SEPARATE,
                                     text_embedding_model=EMBEDDING_MODEL_TYPE)
    texts_df = pipeline.load_data(INPUT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    
    pipeline.index_data(images_dir=IMAGES_DIR, texts=texts, text_summaries=texts, text_filenames=texts_filenames)

    question = "I want to change the behaviour of the stations to continue, if a moderate error occurrs. How can I do this?"
    answer = pipeline.answer_question(question)
    relevant_images = pipeline.rag_chain.retrieved_images
    relevant_texts = pipeline.rag_chain.retrieved_texts
    print("Retrieved images:", len(relevant_images), ", Retrieved texts:", len(relevant_texts))  
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
