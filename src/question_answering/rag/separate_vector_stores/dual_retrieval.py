import os
import uuid
from typing import List, Tuple
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from langchain.retrievers import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.storage import LocalFileStore
#from langchain_nomic.embeddings import NomicEmbeddings
import chromadb.utils.embedding_functions as embedding_functions

from sentence_transformers import SentenceTransformer 
from chromadb import HttpClient
from utils.azure_config import get_azure_config
from rag_env import IMAGES_DIR
from typing import Union, List
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from langchain_core.embeddings import Embeddings
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login
import open_clip        


class SiglipEmbeddings(Embeddings):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", 
                 device: str = "cuda", normalize: bool = True):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.normalize = normalize
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_text(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed_text([text])[0]
    
    def embed_images(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        return self._embed_images(images)
    
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(
            text=texts, 
            padding=True, 
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if self.normalize:
                features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()
    
    def _embed_images(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img))
            else:
                pil_images.append(img)
                
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if self.normalize:
                features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()





class DualSummaryStoreAndRetriever:
    """
    A class providing two separate document stores and vector stores
    to contain texts and images respectively, together with their emebeddings.
    The class also provides a text retriever and an image retriever
    to find documents that are relevant for a query.
    Retrieval is performed using the embeddings in the vector store, but the documents contained in the
    document store are returned. This allows image retrieval via the image summaries, while still ensuring
    that the original images associated with the summaries are returned.
  
    Attributes:
        embedding_model (str): Model used to embed the texts and image summaries.
        store_path (str): Path where the vector and document stores should be saved.
        img_docstore (LocalFileStore): Document store containing images.
        text_docstore (LocalFileStore): Document store containing texts.
        img_vectorstore (Chroma): Vector store containing embedded image summaries.
        text_vectorstore (Chroma): Vector store containing embedded texts.
        img_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for image retrieval.
        text_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for text retrieval.
    """
    def __init__(self, embedding_model, store_path=None, model_id=None):
        if embedding_model == 'openai':
            print("Using text-embedding-3-small")
            azure_embedding_config = get_azure_config()['text_embedding_3']
            self.embeddings = AzureOpenAIEmbeddings(model=azure_embedding_config["model_version"],
                                                    azure_endpoint=azure_embedding_config["openai_endpoint"],
                                                    openai_api_version=azure_embedding_config["openai_api_version"],
                                                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING"),
                                                    chunk_size=64,
                                                    show_progress_bar=True
                                                    )
        elif embedding_model == 'bge':
            model_name = "BAAI/bge-small-en-v1.5"   
            print("Using bge-small-en-1.5 embeddings")
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True, "batch_size": 400}   #, "show_progress_bar":True
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, 
                model_kwargs=model_kwargs, 
                encode_kwargs=encode_kwargs
                )
            

        elif embedding_model == 'all-mini':
            print("Using all-mini embeddings")
            model_name = 'sentence-transformers/all-MiniLM-L12-v2'
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True, "batch_size": 400}     #, "show_progress_bar":True

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name, 
                #model_kwargs=model_kwargs,
                encode_kwargs = encode_kwargs
            )
        elif embedding_model == 'snowflake':
            print("Using Snowflake embeddings")
            model_name="Snowflake/snowflake-arctic-embed-m"
            model_kwargs={ "trust_remote_code": True}   #"device": "cuda",
            encode_kwargs={"normalize_embeddings": True,"prompt_name": "query","batch_size": 256}   #,"show_progress_bar": True
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        else:
            print("ERRRRRORR: NO/ WRONG EMBEDDING MODEL!")
            return "ERRRRRORR: NO/ WRONG EMBEDDING MODEL!"

            
        
        self.store_path = store_path
        img_vectorstore_dir = os.path.join(self.store_path, rf"image_only_{model_id}_vectorstore_{embedding_model}")
        img_docstore_dir = os.path.join(self.store_path, rf"image_only_{model_id}_docstore_{embedding_model}")
        text_vectorstore_dir = os.path.join(self.store_path, rf"text_only_{model_id}_vectorstore_{embedding_model}")
        text_docstore_dir = os.path.join(self.store_path, rf"text_only_{model_id}_docstore_{embedding_model}")
        
        self.img_docstore = LocalFileStore(img_docstore_dir)
        self.text_docstore = LocalFileStore(text_docstore_dir)
        self.id_key = "unique_id"    #doc_id before
        self.doc_ids = []

        self.img_vectorstore = Chroma(
            persist_directory=img_vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"10k_summaries_snowflake_4o_images_new_try" 
        )

        self.text_vectorstore = Chroma(
            persist_directory=text_vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"10k_summaries_snowflake_4o_texts_new_try"
        )
        
        self.is_new_vectorstore = False     #FOR NOW TO GET RESULTS

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {img_vectorstore_dir} already exists")
            #print("lets add to it instead")
            #...

        else:
            print(f"Creating new vectorstore and docstore at path {img_vectorstore_dir}")

        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.text_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 5}
        )
        self.img_retriever = MultiVectorRetriever(
            vectorstore=self.img_vectorstore,
            docstore=self.img_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 5}
        )

        
        
        self.retrieved_docs = []
        self.retrieved_imgs = []
        self.retrieved_texts =[]


    def add_docs(self, doc_summaries: List[str], doc_contents: List[str], doc_filenames: List[str], modality: str):
        doc_ids = doc_filenames
        max_batch_size = 100  
        CHROMA_MAX_BATCH = max_batch_size
        
        seen_ids = set()
        unique_indices = []
        for idx, doc_id in enumerate(doc_ids):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_indices.append(idx)
        
        doc_summaries = [doc_summaries[i] for i in unique_indices]
        doc_contents = [doc_contents[i] for i in unique_indices]
        doc_filenames = [doc_filenames[i] for i in unique_indices]
        doc_ids = [doc_ids[i] for i in unique_indices]
        
        total_docs = len(doc_summaries)
        print(f"Total documents to index after deduplication: {total_docs}")
        
        for i in range(0, total_docs, CHROMA_MAX_BATCH):
            batch_summaries = doc_summaries[i:i+CHROMA_MAX_BATCH]
            batch_contents = doc_contents[i:i+CHROMA_MAX_BATCH]
            batch_filenames = doc_filenames[i:i+CHROMA_MAX_BATCH]
            batch_ids = doc_ids[i:i+CHROMA_MAX_BATCH]
    
            summary_docs = [
                Document(
                    page_content=s,
                    metadata={self.id_key: doc_id, "filename": filename}
                )
                for s, doc_id, filename in zip(batch_summaries, batch_ids, batch_filenames)
            ]
            
            docstore_docs = [
                Document(
                    page_content=content,
                    metadata=(
                        {"filename": filename} | 
                        ({"summary": summary} if modality != "text" else {})
                    )
                ).json().encode("utf-8")
                for content, filename, summary in zip(batch_contents, batch_filenames, batch_summaries)
            ]
            
            if modality == "text":
                self.text_vectorstore.add_documents(summary_docs, batch_size=400)
                self.text_docstore.mset(list(zip(batch_ids, docstore_docs)))
            else:
                self.img_vectorstore.add_documents(summary_docs, batch_size=400)
                self.img_docstore.mset(list(zip(batch_ids, docstore_docs)))


    def retrieve(self, query: str, limit: int, retriever) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve the most relevant documents based on the query.
        """
        if retriever == "image":
            self.retrieved_imgs = self.img_retriever.invoke(query, limit=limit)
            print(self.retrieved_imgs)
        else:
            self.retrieved_texts = self.text_retriever.invoke(query, limit=limit)
            print(self.retrieved_texts)
        return None
        #return self.retrieved_imgs, self.retrieved_texts




class DualClipRetriever:
    """  
    A class providing a vector store to contain multimodal CLIP embeddings of images
    and a vector store and a vector store and associated document store for texts.
    
    The class also provides a text retriever and an image retriever
    to find documents that are relevant for a query.
  
    Attributes:  
        store_path (str): Path where the vector stores and document store should be saved.
        text_model_id:
        text_embedding_model (str): Model used to embed the texts.
        images_dir (str): Directory containing the images to be embedded.
        img_vectorstore (Chroma): Vector store containing images embedded with CLIP.
        text_vectorstore (Chroma): Vector store containing texts embedded with the desired text_embedding_model.
        text_docstore (LocalFileStore): Document store containing texts.
        img_retriever (MultiVectorRetriever): Retriever that used CLIP embeddings for image retrieval.
        text_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for text retrieval.
    """  
    def __init__(self, store_path, text_model_id, text_embedding_model,
                 images_dir=IMAGES_DIR):
        self.images_dir = images_dir

        if text_embedding_model == 'openai':
            print("Using openai embeddings")
            azure_embedding_config = get_azure_config()['text_embedding_3']
            self.embeddings = AzureOpenAIEmbeddings(model=azure_embedding_config["model_version"],
                                                    azure_endpoint=azure_embedding_config["openai_endpoint"],
                                                    openai_api_version=azure_embedding_config["openai_api_version"],
                                                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING"),
                                                    chunk_size=64,
                                                    show_progress_bar=True
                                                    )
        elif text_embedding_model == 'siglip':
            print("lets try siglip")
            model_name= "google/siglip-so400m-patch14-384"    #"google/siglip-base-patch16-384"
            embeddings = SiglipEmbeddings(
                device="cpu",
                normalize=True  
            )

        elif text_embedding_model == 'nomic':
            print("lets try nomic")

            model_name="nomic-ai/nomic-embed-multimodal-7b"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
            )

        elif text_embedding_model == 'newclip':
            print("lets try newclip embeds")

            embeddings = OpenCLIPEmbeddings(model_name="ViT-L-14", checkpoint="laion2b_s32b_b82k")

        elif text_embedding_model == 'bestclip':
            print("lets try the best clip embedding one")
            embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")


            
        else:
            print("Using OPENCLIP smallest embeddings")
                embeddings = OpenCLIPEmbeddings(
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "batch_size": 256, 
                }
            )
            print('loaded the model')            

        self.store_path = store_path
        img_vectorstore_dir = os.path.join(self.store_path, rf"image_only_clip/image_only_vectorstore_clip")
        text_vectorstore_dir = os.path.join(self.store_path, rf"text_only_{text_model_id}/text_only_{text_model_id}_vectorstore_{text_embedding_model}")
        text_docstore_dir = os.path.join(self.store_path, rf"text_only_{text_model_id}/text_only_{text_model_id}_docstore_{text_embedding_model}")

        self.text_docstore = LocalFileStore(text_docstore_dir)
        self.id_key = "doc_id"
        self.doc_ids = []

        
        print('test')


        # Create chroma vectorstore
        self.img_vectorstore = Chroma(
            collection_name="direct_rag_tesing_best_clip_10k_images_test",
            embedding_function=embeddings,
            persist_directory=img_vectorstore_dir
        )


        self.text_vectorstore = Chroma(
            persist_directory=text_vectorstore_dir,
            embedding_function=embeddings,
            collection_name="direct_rag_tesing_best_clip_10k_texts_test"   
        )

        self.is_new_vectorstore = False   #bool(results_img["embeddings"]) or bool(results_text["embeddings"]) 
        
        if self.is_new_vectorstore:
            print(f"Vectorstore at path {img_vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore and docstore at path {img_vectorstore_dir}")

        self.img_retriever = self.img_vectorstore.as_retriever(search_kwargs={"k": 5})

        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.text_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 5}
        )

        self.retrieved_docs = []
        self.retrieved_imgs = []
        self.retrieved_texts =[]


    def add_images(self, images_dir: str = None, batch_size: int = 5400, max_workers: int = 4):
        if not self.is_new_vectorstore and images_dir:
            from pathlib import Path
            from concurrent.futures import ThreadPoolExecutor

            print('test')
            
            image_paths = list(Path(images_dir).rglob("*.jpg"))
            image_paths = image_paths[25000:]    

            print(f"Found {len(image_paths)} images")
    
            existing_files = set() #ghui
            try:
                results = self.img_vectorstore.get(include=["metadatas"])
                existing_files = {meta["filename"] for meta in results["metadatas"] 
                                 if meta and "filename" in meta}
                print(f"Found {len(existing_files)} existing images in vectorstore")
            except Exception as e:
                print(f"Error fetching existing metadata: {e}")
            

            path_dict = {path.name: path for path in image_paths}
            new_images = [path for name, path in path_dict.items() if name not in existing_files]


            unique_paths = new_images
            print(f"After deduplication: {len(unique_paths)} images")

            image_metadatas = []
            for p in unique_paths:
                image_metadatas.append({'filename': p.name})

            for i in range(0, len(unique_paths), batch_size):
                batch_paths = unique_paths[i:i+batch_size]
                
                batch_metadatas = image_metadatas[i:i+batch_size]
                
                self.img_vectorstore.add_images(
                    uris=batch_paths,
                    metadatas=batch_metadatas,
                    batch_size=min(batch_size, 1000)  
                )
                print(f"Added batch {i//batch_size + 1}/{(len(unique_paths)-1)//batch_size + 1}")

            
        else:
            print('Error, check why')
    
    def _process_image_batch(self, paths, metas):
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                images = list(executor.map(self._load_and_preprocess, paths))
    
            docs = [
                Document(
                    page_content=str(img),  
                    metadata=meta
                ) for img, meta in zip(images, metas) if img is not None
            ]
    
            if docs:
                self.img_vectorstore.add_documents(docs, batch_size=400)
                #print(f"Added batch of {len(docs)} images")
    
        except Exception as e:
            print(f"Batch failed: {str(e)}")
    
    def _load_and_preprocess(self, path):
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            
            img = Image.open(path).convert('RGB').resize((224, 224))
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG")   
            
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        except Exception as e:
            print(f"Failed {path.name}: {str(e)}")
            return None
    
    def extract_image_uris(self, root_path: str, image_extension: str = ".jpg") -> List[str]:
        from pathlib import Path
        return [str(p) for p in Path(root_path).rglob(f"*{image_extension}")]

    def extract_manual_name(self, uri: str) -> str:  
        # Split the URI into parts  
        parts = uri.split('/')
        # The directory name is the second to last element  
        directory_name = parts[-1]
        # Append ".pdf" to the directory name  
        return directory_name



    def add_texts(self, doc_summaries: List[str], doc_contents: List[str], doc_filenames: List[str]):
        """    
        Add texts to the text vector store and document store with batch processing.
        
        :param doc_summaries: Text summaries for vector store
        :param doc_contents: Original texts for document store
        :param doc_filenames: Filenames used as unique IDs and metadata
        """
        if True:    #not self.is_new_vectorstore:
            print("Adding documents...")
            CHROMA_MAX_BATCH = 5400  
            BATCH_SIZE = 400  
            
            seen_ids = set()
            unique_indices = []
            for idx, filename in enumerate(doc_filenames):
                if filename not in seen_ids:
                    seen_ids.add(filename)
                    unique_indices.append(idx)
            
            doc_summaries = [doc_summaries[i] for i in unique_indices]
            doc_contents = [doc_contents[i] for i in unique_indices]
            doc_filenames = [doc_filenames[i] for i in unique_indices]
            doc_ids = doc_filenames  
            
            total_docs = len(doc_summaries)
            print(f"Total documents to index after deduplication: {total_docs}")
            
            for i in range(0, total_docs, CHROMA_MAX_BATCH):
                batch_summaries = doc_summaries[i:i+CHROMA_MAX_BATCH]
                batch_contents = doc_contents[i:i+CHROMA_MAX_BATCH]
                batch_filenames = doc_filenames[i:i+CHROMA_MAX_BATCH]
                batch_ids = doc_ids[i:i+CHROMA_MAX_BATCH]
                
                summary_docs = [
                    Document(
                        page_content=s,
                        metadata={self.id_key: doc_id, "filename": filename}
                    )
                    for s, doc_id, filename in zip(batch_summaries, batch_ids, batch_filenames)
                ]
                
                docstore_docs = [
                    Document(
                        page_content=content,
                        metadata={"filename": filename}
                    ).json().encode("utf-8")
                    for content, filename in zip(batch_contents, batch_filenames)
                ]
                
                self.text_vectorstore.add_documents(summary_docs, batch_size=BATCH_SIZE)
                self.text_docstore.mset(list(zip(batch_ids, docstore_docs)))

                print(f'added {i} batches')
                
            print(f"Successfully added {total_docs} documents in batches")
        else:
            print("Documents already added, skipping...")


    def retrieve(self, query: str, limit: int, retriever) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve the most relevant documents based on the query.
        """
        if retriever == "image":
            self.retrieved_imgs = self.img_retriever.invoke(query, limit=limit)
        else:
            self.retrieved_texts = self.text_retriever.invoke(query, limit=limit)

        return self.retrieved_imgs, self.retrieved_texts
    
