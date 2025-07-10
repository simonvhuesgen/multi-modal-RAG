import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from data_summarization.context_summarization import ImageSummarizer, TextSummarizer
from langchain_openai import AzureChatOpenAI
from question_answering.rag.separate_vector_stores.dual_rag_chain import DualMultimodalRAGChain
from question_answering.rag.separate_vector_stores.dual_retrieval import DualSummaryStoreAndRetriever
from utils.azure_config import get_azure_config
from langchain_core.documents import Document
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import EMBEDDING_MODEL_TYPE, IMG_SUMMARIES_CACHE_DIR, INPUT_DATA, MODEL_TYPE, TEXT_SUMMARIES_CACHE_DIR, VECTORSTORE_PATH_SUMMARIES_SEPARATE


class DualMultimodalRAGPipelineSummaries:
    """
    Initializes the Multimodal RAG pipeline with separate vector stores and retrievers for texts and images.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Transforms images into textual summaries and embeds the image summaries.
    Image summaries and texts are stored in separate vector stores.
    
    Attributes:
        model_type (str): Type of the model to use for answer synthesis.
        store_path (str): Path to the directory where the vector database is stored.
        embedding_model (str): Text embedding model used to embed texts and image summaries.
        model: The model used for answer synthesis loaded based on `model_type`.
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        image_summarizer: (ImageSummarizer): Used to generate textual summaries from images.
        sotre_and_retriever (SummaryStoreAndRetriever): Retrieval using textual summaries from images.
        rag_chain (MultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path, embedding_model):
        
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
        #self.image_summarizer = ImageSummarizer(self.model, self.tokenizer)
        self.store_and_retriever = DualSummaryStoreAndRetriever(embedding_model=embedding_model,
                                                            store_path=store_path,
                                                            model_id=model_type)
        print('loaded the store and retriever')
        self.rag_chain = DualMultimodalRAGChain(self.model,
                                            self.tokenizer,
                                            self.store_and_retriever.text_retriever,
                                            self.store_and_retriever.img_retriever)
        print('loaded the chain')

    def load_data(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(path)
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        img = df[["doc_id", "image_bytes"]]
        images = img.dropna(subset=['image_bytes'])  
        return texts, images


    def load_new_df(self, input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = input_df["type"] == "text"
        texts = input_df[mask]
        images = input_df[~mask]
    
        text_df = pd.DataFrame({
            "text": texts["text_content"],
            "doc_id": texts["doc_id"],
            "unique_id": texts["doc_id"].astype(str) + "_" + texts["chunk_index"].astype(str)
        }).reset_index(drop=True)
    
        if not images.empty:
            paths = images["original_image_path"].str.replace(
                r"^s3://.*/", "data/399k_imgs/", regex=True
            )
            full_paths = "../../" + paths
            
            exists_mask = [Path(fp).exists() for fp in full_paths]
            valid_images = images[exists_mask]
            valid_paths = full_paths[exists_mask]
            
            image_bytes = [Path(fp).read_bytes() for fp in valid_paths]
            
            image_df = pd.DataFrame({
                "image_bytes": image_bytes,
                "doc_id": valid_images["doc_id"].values,
                "unique_id": valid_images["original_image_path"].str.split("/").str[-1].values,
                "image_summary": valid_images["image_summary"].values
            }).reset_index(drop=True)
        else:
            image_df = pd.DataFrame()
    
        return text_df, image_df

    """
    def load_new_df(self, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        
        return pd.DataFrame(text_records), pd.DataFrame(image_records)
    
    """
    def summarize_data(self, texts: List[str], images: List[bytes], cache_file:str) -> Tuple[List[str], List[str], List[str]]:
        # Summarize images
        img_base64_list, image_summaries = self.image_summarizer.summarize(images, cache_file)
        # Summarize texts
        #if texts:
            #text_summaries = self.text_summarizer.summarize(texts)
            #return text_summaries, image_summaries, img_base64_list
        return texts, image_summaries, img_base64_list

    
    def index_data(self, texts: List[str]=None, text_summaries: List[str]=None,
                   image_summaries: List[str]=None, images_base64: List[str]=None,
                   text_filenames: List[str]=None, image_filenames: List[str]=None):
        print("Adding texts to store")
        self.store_and_retriever.add_docs(texts, texts, text_filenames, "text")
        if image_summaries:
            print("Adding image summaries to store")
            #print(image_summaries)
            self.store_and_retriever.add_docs(image_summaries, images_base64, image_filenames, "image")
    

    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = DualMultimodalRAGPipelineSummaries(model_type=MODEL_TYPE,
                                     store_path=VECTORSTORE_PATH_SUMMARIES_SEPARATE,
                                     embedding_model=EMBEDDING_MODEL_TYPE)

    texts_df, images_df = pipeline.load_data(INPUT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = pipeline.image_summarizer.summarize(images, IMG_SUMMARIES_CACHE_DIR)

    pipeline.index_data(texts,
                        image_summaries=image_summaries,
                        images_base64=img_base64_list,
                        image_filenames=image_filenames,
                        text_filenames=texts_filenames)

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
