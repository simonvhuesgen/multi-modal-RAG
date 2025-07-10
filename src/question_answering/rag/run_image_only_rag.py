import os
import pandas as pd
from utils.base64_utils.base64_utils import encode_image_from_bytes
from single_vector_store.rag_pipeline_clip import MultimodalRAGPipelineClip
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from rag_env import EMBEDDING_MODEL_TYPE, IMAGES_DIR, IMG_SUMMARIES_CACHE_DIR, INPUT_DATA, MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA, VECTORSTORE_PATH_IMAGE_ONLY, VECTORSTORE_PATH_IMAGE_ONLY_testing

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

def write_to_df(df, user_query, reference_answer, generated_answer, context, image, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image]
    df.to_json(output_file, orient="records", indent=2)


def process_dataframe(input_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image']
        output_df= pd.DataFrame(columns=columns)
    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")
        user_query = input_df["question"][index]
        print("USER QUERY:\n", user_query)
        reference_answer = input_df["reference_answer"][index]
        print("REFERENCE ANSWER:", reference_answer)
        generated_answer = pipeline.answer_question(user_query)
        print("GENERATED ANSWER:\n", generated_answer)
        relevant_images = pipeline.rag_chain.retrieved_images
        print("Retrieved images:", len(relevant_images))
        image = relevant_images[0] if len(relevant_images) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, [], image, output_file)
    return output_df


def run_pipeline_with_clip(model, vectorstore_path, images_dir, reference_qa, output_dir):
    pipeline = MultimodalRAGPipelineClip(model_type=model, store_path=vectorstore_path)
    # the pipeline is indexed only with images, no texts are added
    pipeline.index_data(images_dir=images_dir, texts_df=None)
    
    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_image_only_clip.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df


def new_pipe_clip(model, vectorstore_path, images_dir, reference_qa, output_dir):
    
    pipeline = MultimodalRAGPipelineClip(model_type=model, store_path=vectorstore_path)
    # the pipeline is indexed only with images, no texts are added
    pipeline.index_data(images_dir=images_dir, texts_df=None)
    
    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_image_only_clip.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df
    
    
def run_pipeline_with_summaries(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = MultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path,
                                                        embedding_model=embedding_model)
    
    _, images_df = summaries_pipeline.load_data(input_df)
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    # the pipeline is indexed only with images, no texts are added
    summaries_pipeline.index_data(image_summaries=image_summaries,
                                  images_base64=img_base64_list, image_filenames=image_filenames,
                                  texts = None, text_summaries=None)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_image_only_summaries.json")
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
            print(type(image_bytes), "and    :    ", image_bytes[:20])
            continue

    return img_base64_list

def new_pipe_summaries(qa_model, embedding_model, vectorstore_path, reference_qa, output_dir):
    input_df = pd.read_csv("../../very_small_df.csv")
    summaries_pipeline = MultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path,
                                                        embedding_model=embedding_model)
    
    _, images_df = summaries_pipeline.load_new_df(input_df)
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    # the pipeline is indexed only with images, no texts are added
    img_base64_list = get_base_ersatz(images)
    image_summaries =  images_df[["image_summary"]]["image_summary"].tolist()
    
    summaries_pipeline.index_data(image_summaries=image_summaries,
                                  images_base64=img_base64_list, image_filenames=image_filenames,
                                  texts = None, text_summaries=None)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_image_only_summaries.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df
  
if __name__ == "__main__":  
    # uncomment one of the following two options to run image-only RAG either with CLIP embedings or with image summaries
    #rag_results_clip = run_pipeline_with_clip(model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
     #                                    images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR)

    #new_pipe_clip(model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY_testing,
     #                                    images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR)
    
    #rag_results_summaries = run_pipeline_with_summaries(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
     #                                         embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
      #                                        reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
       #                                       img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)

    new_pipe_summaries(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY_testing,
                                              embedding_model=EMBEDDING_MODEL_TYPE,
                                              reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR
                                              )