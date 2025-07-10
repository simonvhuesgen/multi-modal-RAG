import io
from evaluation.evaluators.base_evaluator import BaseEvaluator
from PIL import Image
from utils.base64_utils.base64_utils import decode_image_to_bytes

"""
Contains evaluator classes for specific metrics with LLaVA as evaluator model.
The required input arguments vary depending on the metric to be evaluated.
Each Evaluator inherits from the BaseEvaluator and implements the get_prompt method for prompt construction.
Each prompt instructs the model to evaluate the desired metric, it provides a description of the metric,
it provides the required input arguments to the model, and it describes the output format required.
"""
    

class ImageContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, image: str, model, tokenizer) -> dict:
        super().__init__(model=model, user_query=user_query, image=image, tokenizer=tokenizer)
        
    def get_prompt(self, inputs: dict):
        
        if not inputs["image"]:
            return None
        
        json_format = "{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] {'<image>' if inputs['image'] else ' '}\n
            Evaluate the following metric by comparing the user query with the provided image:\n
            image_context_relevancy: Is the content of the image relevant to the user\'s query "{inputs["user_query"]}", i.e. can it contribute to answer the query? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        image = decode_image_to_bytes(inputs['image'])
        image = Image.open(io.BytesIO(image))
        
        return {"prompt": prompt, "image": image}


class ImageFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, image: str, model, tokenizer) -> dict:
        super().__init__(user_query=user_query, generated_answer=generated_answer, image=image, model=model, tokenizer=tokenizer)

    def get_prompt(self, inputs: dict):
        
        if not inputs["image"]:
            return None
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] {'<image>' if inputs['image'] else ' '}\n
            Evaluate the following metric by comparing the answer with the provided image:\n
            image_faithfulness: Is the answer faithful to the content of the image, i.e. does it factually align with the image? (YES or NO)\n
            ANSWER: "{inputs["generated_answer"]}"\n\
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """

        image = decode_image_to_bytes(inputs['image'])
        image = Image.open(io.BytesIO(image))
        
        return {"prompt": prompt, "image": image}
    
    
class ContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, context: str, image: str, model, tokenizer) -> dict:
        super().__init__(model=model, user_query=user_query, context=context, image=image, tokenizer=tokenizer)
        
    def get_prompt(self, inputs: dict):
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] Evaluate the following metric by comparing the user query with the provided image and text:\n\n
            context_relevancy: Is the context provided (as text and/or image) relevant to the user\'s query? (YES or NO)\n
            USER QUERY: {inputs["user_query"]}\n
            {"TEXT: " + inputs["context"] if inputs["context"] else ""}
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        return {"prompt": prompt}


class TextContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, context: str, model, tokenizer) -> dict:
        super().__init__(model=model, user_query=user_query, context=context, tokenizer=tokenizer)
        
        
    def get_prompt(self, inputs: dict):
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] Evaluate the following metric:\n
            text_context_relevancy: Is the text "{inputs["context"]}" relevant to the user\'s query "{inputs["user_query"]}"? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        return {"prompt": prompt}


class AnswerRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, model, tokenizer) -> dict:
        super().__init__(model=model, user_query=user_query, generated_answer=generated_answer, tokenizer=tokenizer)
        
    
    def get_prompt(self, inputs: dict):
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] Evaluate the following metric:\n
            answer_relevancy: Is the answer "{inputs["generated_answer"]}" relevant to the user\'s query "{inputs["user_query"]}"? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        return {"prompt": prompt}


class AnswerCorrectnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, reference_answer: str, model, tokenizer) -> dict:
        super().__init__(model=model, user_query=user_query, reference_answer=reference_answer,
                         generated_answer=generated_answer, tokenizer=tokenizer)
         
    def get_prompt(self, inputs: dict):
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] You are given a question, the correct reference answer, and the student\'s answer. \
            You are asked to grade the student\'s answer as either correct or incorrect, based on the reference answer. \
            Ignore differences in punctuation and phrasing between the student answer and true answer. \
            It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.\
            USER QUERY: "{inputs["user_query"]}"\n\
            REFERENCE ANSWER: "{inputs["reference_answer"]}"\n\
            STUDENT ANSWER: "{inputs["generated_answer"]}"\n\
            answer_correctness: Is the student's answer correct? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        return {"prompt": prompt}


class TextFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, context: str, model, tokenizer) -> dict:
        super().__init__(user_query=user_query, generated_answer=generated_answer, context=context, model=model, tokenizer=tokenizer)
        
    def get_prompt(self, inputs: dict):
        
        json_format = "\{grade: '', 'reason': ''}"
        
        prompt = f"""
            [INST] Evaluate the following metric:\n
            text_faithfulness: Is the answer faithful to the context provided by the text, i.e. does it factually align with the context? (YES or NO)\n
            ANSWER: "{inputs["generated_answer"]}"\n\
            TEXT: "{inputs["context"]}"\n\
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
            {json_format}
            Output the reason as a string, not as a list.
            The only allowed grades are 1 or 0. [/INST]
            """
        
        return {"prompt": prompt}
