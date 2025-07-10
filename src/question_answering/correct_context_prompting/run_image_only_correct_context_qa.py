import os
import pandas as pd
from utils.base64_utils.base64_utils import encode_image_from_bytes
from correct_context_qa_chain import CorrectContextQAChain
from rag_env import INPUT_DATA, MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA

def write_to_df(df, user_query, reference_answer, generated_answer, context, image, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image]
    df.to_json(output_file, orient="records", indent=2)
    
    
def process_dataframe(input_df, qa_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image']
        output_df= pd.DataFrame(columns=columns)
    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")
        question = qa_df["question"][index]
        print("QUESTION:", question)
        reference_answer = qa_df["reference_answer"][index]
        print("REFERENCE:", reference_answer)
        context = input_df["text"][index]
        img_bytes = input_df["image_bytes"][index]
        image = encode_image_from_bytes(img_bytes)
        inputs = dict()
        inputs["context"] = dict()
        inputs["context"]["images"] = [image]
        inputs["context"]["texts"] = []
        inputs["question"] = question
        generated_answer = pipeline.run(inputs)
        print("GENERATED ANSWER:", generated_answer)
        write_to_df(output_df, question, reference_answer, generated_answer, context, image, output_file)
    return output_df


if __name__ == "__main__":

    chain = CorrectContextQAChain(model_type=MODEL_TYPE)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_image_only_correct_context.json")
    # dataframe containing the questions and reference answers
    qa_df = pd.read_excel(REFERENCE_QA)
    # dataframe containing the extracted texts and images for each qa pair
    input_df = pd.read_parquet(INPUT_DATA)
    process_dataframe(input_df, qa_df, chain, output_file)