import os  
import pandas as pd
import time
import base64
from pathlib import Path
from huggingface_hub import login
from single_vector_store.rag_pipeline_clip import MultimodalRAGPipelineClip
from separate_vector_stores.dual_rag_pipeline_clip import DualMultimodalRAGPipelineClip
from utils.base64_utils.base64_utils import *
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from separate_vector_stores.dual_rag_pipeline_summaries import DualMultimodalRAGPipelineSummaries
from rag_env import *


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


def write_to_df(df, user_query, reference_answer, generated_answer, context, image, unique_id_text, unique_id_img, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image, unique_id_text, unique_id_img]
    df.to_json(output_file, orient="records", indent=2)


def process_dataframe(input_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image', 'txt_ids', 'unique_id_img']
        output_df= pd.DataFrame(columns=columns)
    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")
        user_query = input_df["question"][index]
        reference_answer = input_df["answer"][index]    
        generated_answer = pipeline.answer_question(user_query)
        relevant_images, img_ids = pipeline.rag_chain.retrieved_images
        relevant_texts, txt_ids = pipeline.rag_chain.retrieved_texts
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []




        
        image = relevant_images[0] if len(relevant_images) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, txt_ids, img_ids, output_file)
    return output_df



def run_pipeline_with_clip_single(model, input_df, vectorstore_path, images_dir, reference_qa, output_dir):
    pipeline = MultimodalRAGPipelineClip(model_type=model, store_path=vectorstore_path)
    texts_df = pipeline.load_data(input_df)
    pipeline.index_data(texts_df=texts_df, images_dir=images_dir)

    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_multimodal_clip_single.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df



def run_pipeline_with_clip_dual(model, input_df, vectorstore_path, images_dir, reference_qa, output_dir, text_embedding_model):
    pipeline = DualMultimodalRAGPipelineClip(model_type=model,
                                             store_path=vectorstore_path,
                                             text_embedding_model=text_embedding_model)
    texts_df = pipeline.load_data(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    
    pipeline.index_data(images_dir=images_dir, texts=texts, text_summaries=texts, text_filenames=texts_filenames)

    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_multimodal_clip_dual.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df

def new_pipe_clip(model, vectorstore_path, images_dir, reference_qa, output_dir, text_embedding_model):

    input_df =  pd.read_csv("../../data/tenk_subset.csv")  

    input_df = input_df.iloc[60000:] 

    print("lets try for: ",input_df.shape,"data points")

    print('get pipeline')
    
    pipeline = DualMultimodalRAGPipelineClip(model_type=model,
                                             store_path=vectorstore_path,
                                             text_embedding_model=text_embedding_model)

    print('got pipeline')
    
    texts_df, _ = pipeline.load_new_df(input_df)
    print(texts_df.shape)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["unique_id"]]["unique_id"].tolist()

    print('start indexing')

    start_time = time.time()
    
    pipeline.index_data(images_dir=images_dir, texts=texts, text_summaries=texts, text_filenames=texts_filenames)

    end_time = time.time()

    print('done')

    print('IT TOOK ',end_time - start_time, ' SECONDS!')


    text_count = pipeline.dual_retriever.text_vectorstore._collection.count()
    img_count = pipeline.dual_retriever.img_vectorstore._collection.count()
    print(f"Vectorstore counts - Texts: {text_count}, Images: {img_count}")

    df = pd.read_excel(reference_qa)

    output_file = os.path.join(output_dir, f"rag_output_{model}_multimodal_clip_dual_test_BEST_CLIP.json")
    output_df = process_dataframe(df, pipeline, output_file)
    
    return output_df
    

    
    
def run_pipeline_with_summaries_single(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = MultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path,
                                                        embedding_model=embedding_model)
    
    texts_df, images_df = summaries_pipeline.load_data(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_summaries_single.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df  



def run_pipeline_with_summaries_dual(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = DualMultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path, 
                                                        embedding_model=embedding_model)

    texts_df, images_df = summaries_pipeline.load_data(input_df)

    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()

    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()

    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)

    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_summaries_dual.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df  


def get_base_ersatz(image_bytes_list):
    img_base64_list = []
    for i, image_bytes in enumerate(image_bytes_list):
        try:
            img_base64 = encode_image_from_bytes(image_bytes)
            img_base64_list.append(img_base64)
        except:
            print(f"Failed to encode img {i}...")
            continue

    return img_base64_list



def lets_do_indexing(qa_model, embedding_model, vectorstore_path, input_df, img_summaries_dir):
    print("loading csv df")
    input_df = pd.read_csv("../../testing_df_data_9k.csv")
    input_df = input_df.iloc[:1000000]
    print("done loading")
    print("lets try for: ",input_df.shape,"data points")

    print("DATASET ready")
    pipeline = DualMultimodalRAGPipelineSummaries(
        model_type=qa_model,
        store_path=vectorstore_path,
        embedding_model=embedding_model
    )
    print("pipeline init done")
    print("loading data1")
    texts_df, images_df = pipeline.load_new_df(input_df)
    print("loading data1 done")
    print(texts_df.columns, images_df.columns)
    print("loading data2 done")
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["unique_id"]]["unique_id"].tolist()     #currently testing
    print("loading data3 done")
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["unique_id"]]["unique_id"].tolist()
    print("loading data4 done")
    img_base64_list = get_base_ersatz(images)
    print("loading data5 done")
    image_summaries =  images_df[["image_summary"]]["image_summary"].tolist()  #summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    print("loading data6")
    
    print("start to index: ", input_df.shape)
    
    start_time = time.time()

    pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    end_time = time.time()
    execution_time = end_time - start_time
    # Verify indexing
    text_count = pipeline.store_and_retriever.text_vectorstore._collection.count()
    img_count = pipeline.store_and_retriever.img_vectorstore._collection.count()
    print(f"Vectorstore counts - Texts: {text_count}, Images: {img_count}")

    print(start_time, end_time,"done in: ", execution_time)

    
    return "testing"



def new_pipe(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    print("loading csv df")
    input_df =  pd.read_csv("../../data/tenk_subset_4o_fixed.csv")   #("../../data/tenk_subset_4o_fixed.csv") #_4o_fixed    
    print("lets try for: ",input_df.shape,"data points")


    pipeline = DualMultimodalRAGPipelineSummaries(
        model_type=qa_model,
        store_path=vectorstore_path,
        embedding_model=embedding_model
    )
    print("pipeline init done")
    print("loading data1")
    
    texts_df, images_df = pipeline.load_new_df(input_df)
    print(texts_df.columns, images_df.columns)
    print("loading data1 done")

    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["unique_id"]]["unique_id"].tolist() 

    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["unique_id"]]["unique_id"].tolist()
    print("loading data3 done")

    img_base64_list = get_base_ersatz(images)
    image_summaries =  images_df[["image_summary"]]["image_summary"].tolist()  #summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    
    print("start to index: ", input_df.shape)
    
    start_time = time.time()

    pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    end_time = time.time()
    execution_time = end_time - start_time
    
    text_count = pipeline.store_and_retriever.text_vectorstore._collection.count()
    img_count = pipeline.store_and_retriever.img_vectorstore._collection.count()
    print(f"Vectorstore counts - Texts: {text_count}, Images: {img_count}")

    print(start_time, end_time,"done in: ", execution_time)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_summaries_dual_snowflake_4o_new_try.json")
    
    return process_dataframe(df, pipeline, output_file)



  
if __name__ == "__main__":  
    # uncomment one of the following options to run multimodal RAG either with CLIP embedings or with image summaries
    # and either with a single vector store for both modalities or a dedicated one for each modality.
    #rag_results_clip_single = run_pipeline_with_clip_single(model=MODEL_TYPE, input_df=INPUT_DATA,
     #                                                       vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
      #                                                      images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA,
       #                                                     output_dir=RAG_OUTPUT_DIR)
    
    #rag_results_clip_dual = run_pipeline_with_clip_dual(model=MODEL_TYPE, input_df=INPUT_DATA,
     #                                                  vectorstore_path=VECTORSTORE_PATH_CLIP_SEPARATE,
      #                                                  images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA, 
       #                                                 output_dir=RAG_OUTPUT_DIR, text_embedding_model=EMBEDDING_MODEL_TYPE)

    #new_pipe_clip(model = MODEL_TYPE, vectorstore_path = VECTORSTORE_PATH_CLIP_SEPARATE_testing, images_dir=IMAGES_DIR , reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR, text_embedding_model=EMBEDDING_MODEL_TYPE)
    
    #rag_results_summaries_single = run_pipeline_with_summaries_single(qa_model=MODEL_TYPE,
     #                                                                 vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SEPARATE,
      #                                                                embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
       #                                                               reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
        #                                                              img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)
    
    new_pipe(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SEPARATE_testing,
                                                                    embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
                                                                    reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
                                                                    img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)

    #lets_do_indexing(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SEPARATE_testing,
     #                                                               embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
      #                                                              img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)
    
    #rag_results_summaries_dual = run_pipeline_with_summaries_dual(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SEPARATE,
     #                                                               embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
      #                                                              reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
       #                                                             img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)