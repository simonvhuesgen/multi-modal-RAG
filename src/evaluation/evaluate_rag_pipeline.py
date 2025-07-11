import os
import numpy as np
import json
import pandas as pd
from typing import List
from evaluation_module import EvaluationModule     



METRICS = ['Answer Correctness', 'Answer Relevancy']
IMAGE_METRICS = ['Image Faithfulness', 'Image Context Relevancy']
TEXT_METRICS = ['Text Faithfulness', 'Text Context Relevancy']
AGGREGATED_METRICS = ['Faithfulness', 'Context Relevancy']



def evaluate_row(metrics: List[str], index: int, context: str, image: str, user_query: str, generated_answer: str,
                 reference_answer: str, evaluator: str, scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the results for one output of a RAG system.

    :param metrics: A list of metrics to be evaluated.
    :param index: Index of the current row in the dataframe.
    :param context: Textual context retrieved by the retrieval system.
    :param image: Base64 encoded string of an image retrieved by the retrieval system.
    :param user_query: Question to be answered by the RAG system.
    :param generated_answer: Answer generated by the RAG system.
    :param reference_answer: Gold standard answer to the user query.
    :param evaluator: Model to be used as evaluator (should be gpt4_vision or llava).
    :param scores_df: A dataframe containing the evaluation results.
    
    :return: A dataframe containing the evaluation results.
    """
    print(f"Image type: {type(image)}, Length: {len(image) if image else 0}")
    #print("heeeeeere:  ",metrics,"and",user_query,"and",context,"and",image,"and",generated_answer)
    try:
        results = evaluator.evaluate(metrics=metrics,
                                    query=user_query,
                                    context=context,
                                    image=image,
                                    generated_answer=generated_answer,
                                    reference_answer=reference_answer)
    except:
        print("failed to evaluate image from query: ",user_query)
        return None



    for metric_name, result in results.items():
        if hasattr(result, 'invoke'):
            result = result.invoke({})
            
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"grade": 0, "reason": "Invalid format"}

        grade = result.get("grade", 0)
        reason = result.get("reason", "No evaluation provided")

        scores_df.at[index, f"{metric_name} grade"] = grade
        scores_df.at[index, f"{metric_name} reason"] = reason
    
    
    return scores_df


def handle_no_data(index, data_type: str, scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Defines behaviour for cases where no text or image is provided as context.
    In this case the grade will be None and the reason is, that no data has been provided.

    :param metrics: A string indicating the missing data type (text or image).
    :param scores_df: A dataframe containing the evaluation results.
    
    :return: A dataframe containing the evaluation results.
    """  
    scores_df.at[index, f"{data_type} Faithfulness grade"] = None
    scores_df.at[index, f"{data_type} Faithfulness reason"] = f"No {data_type} provided"
    scores_df.at[index, f"{data_type} Context Relevancy grade"] = None
    scores_df.at[index, f"{data_type} Context Relevancy reason"] = f"No {data_type} provided"
    return scores_df


def evaluate_dataframe(input_df: pd.DataFrame, evaluator: str, output_file: str) -> pd.DataFrame:
    """
    Evaluates the outputs of a RAG system.

    :param input_df: Dataframe containing the outputs of a RAG system.
    :param evaluator: Model to be used as evaluator (should be gpt4_vision, or llava).
    :param output_file: json file where the evaluation results are stored.
    
    :return: A dataframe containing the evaluation results.
    """
    scores_df = pd.DataFrame()
    for index, row in input_df.iterrows():
        print(f"Evaluating query no. {index+1}...")
        user_query = input_df["user_query"][index]
        reference_answer = input_df["reference_answer"][index]
        generated_answer = input_df["generated_answer"][index]
        context = input_df["context"][index]
        image = input_df["image"][index] if input_df["image"][index] else []    
        
            
        metrics = METRICS.copy()
        if not image:
            scores_df = handle_no_data(index, "Image", scores_df)
        else:
            metrics.extend(IMAGE_METRICS)
        if not context:
            scores_df = handle_no_data(index, "Text", scores_df)
        else:
            metrics.extend(TEXT_METRICS)

        scores_df = evaluate_row(metrics, index, context, image, user_query, generated_answer, reference_answer, evaluator, scores_df)
                
        for metric in AGGREGATED_METRICS:
            img_metric = scores_df.at[index, f"Image {metric} grade"]
            text_metric = scores_df.at[index, f"Text {metric} grade"]
        
            df_tmp = pd.DataFrame()


            if img_metric == None:
                img_metric = -1
            if text_metric == None:
                text_metric = -1
            df_tmp[metric] = [int(img_metric), int(text_metric)]
            grade = df_tmp[metric].mean()
            scores_df.at[index, f"{metric} grade"] = grade
        
        scores_df.to_json(output_file, orient="records", indent=2)
    return scores_df
  
  
def calculate_and_print_averages(scores_df: pd.DataFrame):
    """
    Averages the scores for each metric over the entire dataset and prints them.
    :param scores_df: Dataframe containing the evaluation results.
    """
    #average_dict = {}
    #for grade in METRICS + IMAGE_METRICS + TEXT_METRICS + AGGREGATED_METRICS:
    #    average_dict[grade] = int(scores_df[f'{grade} grade']).mean()
    #    print(f"{grade.capitalize()}: {average_dict[grade]}")
    print(scores_df.to_string())


if __name__ == "__main__":
    generator_model = "gpt4o"   # model that was used as generator in the rag pipeline to be evaluated
    evaluator_model = "gpt4o"   # choose among llava and gpt4_vision

    #rag_output_file =  "sample_data/rag_outputs/testing_new_enw/rag_output__multimodal_clip_dual.json"
    #evaluation_output_file = "sample_data/rag_outputs/test_10k_imgs.json"

    
    # json file containing the results of a rag pipeline
    rag_output_file = rf"sample_data/rag_outputs/rag_output_img_summaries_{generator_model}.json"
    # file for saving evaluaton results
    evaluation_output_file = rf"sample_data/rag_evaluation_results/evaluation_{generator_model}_generator_{evaluator_model}_evaluator.json"

    evaluator = EvaluationModule(evaluator_model)

    input_df = pd.read_json(rag_output_file)

    print(input_df.columns)

    
    scores_df = evaluate_dataframe(input_df, evaluator, evaluation_output_file)
    calculate_and_print_averages(scores_df)