import os
import pandas as pd  
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from rag_env import EMBEDDING_MODEL_TYPE, INPUT_DATA, MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA, VECTORSTORE_PATH_TEXT_ONLY, VECTORSTORE_PATH_TEXT_ONLY_testing


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
        relevant_texts = pipeline.rag_chain.retrieved_texts
        print("Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, [], output_file)
    return output_df


  
if __name__ == "__main__":
    input_df = pd.read_csv("../../very_small_df.csv")
    
    pipeline = MultimodalRAGPipelineSummaries(MODEL_TYPE, VECTORSTORE_PATH_TEXT_ONLY_testing, EMBEDDING_MODEL_TYPE)
    texts_df, _ = pipeline.load_new_df(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    # the pipeline is indexed only with text, no images are added
    pipeline.index_data(texts=texts, text_filenames=texts_filenames)
    df = pd.read_excel(REFERENCE_QA)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_text_only.json")
    output_df = process_dataframe(df, pipeline, output_file)