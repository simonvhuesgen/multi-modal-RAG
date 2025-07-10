"""
Configuration for the RAG pipeline. Replace the paths with your own actual paths.
The data used here is small sample data to show the expected format, not the real data.
This toy data will not lead to good results.
"""

# model to use for answer synthesis and image summarization. ('gpt4_vision', otherwise LLaVA will be used)
MODEL_TYPE = ''
# text embedding model, set to 'openai' to use text-embeding-3-small, otherwise bge-m3 will be used
EMBEDDING_MODEL_TYPE = 'snowflake'

# excel file containing questions and reference answers
REFERENCE_QA =  "dataset/thesis_450_set.xlsx"

# parquet file where extracted texts and image bytes are stored if you want to use the original pipelone with pdf extracted image and text
INPUT_DATA = r'sample_data/extracted_texts_and_imgssssssss.parquet'    

# directory where extracted images are stored
IMAGES_DIR = r"images"   #images

# directories where vector stores are saved
VECTORSTORE_PATH_CLIP_SINGLE = r"sample_data/vec_and_doc_stores/clip/new"
VECTORSTORE_PATH_CLIP_SEPARATE = r"sample_data/vec_and_doc_stores/clip_dual/new"
VECTORSTORE_PATH_SUMMARIES_SINGLE = r"sample_data/vec_and_doc_stores/image_summaries/new"
VECTORSTORE_PATH_SUMMARIES_SEPARATE = r"sample_data/vec_and_doc_stores/image_summaries_dual/new"


VECTORSTORE_PATH_IMAGE_ONLY = r"sample_data/vec_and_doc_stores/image_only/new"
VECTORSTORE_PATH_TEXT_ONLY = r"sample_data/vec_and_doc_stores/text_only/new"

VECTORSTORE_PATH_SUMMARIES_SEPARATE_testing =  r"sample_data/vec_and_doc_stores/image_summaries_dual/new_with_figure_captions"
VECTORSTORE_PATH_CLIP_SEPARATE_testing =  r"sample_data/vec_and_doc_stores/image_summaries_dual/new"

VECTORSTORE_PATH_TEXT_ONLY_testing = r"sample_data/vec_and_doc_stores/text_only/new"
VECTORSTORE_PATH_IMAGE_ONLY_testing = r"sample_data/vec_and_doc_stores/image_only/new"


# directory where the output of a RAG pipeline is stored
RAG_OUTPUT_DIR = r"sample_data/rag_outputs/new"

# directory where the evaluation results for a RAG pipeline are stored
EVAL_RESULTS_PATH = r"sample_data/new_eval_results"
