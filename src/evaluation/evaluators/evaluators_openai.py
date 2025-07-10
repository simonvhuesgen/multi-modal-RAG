from langchain_core.messages import HumanMessage
from evaluation.evaluators.base_evaluator import BaseEvaluator

"""
Contains evaluator classes for specific metrics with GPT-4V(ision) as evaluator model.
The required input arguments vary depending on the metric to be evaluated.
Each Evaluator inherits from the BaseEvaluator and implements the get_prompt method for prompt construction.
Each prompt instructs the model to evaluate the desired metric, it provides a description of the metric,
it provides the required input arguments to the model, and it describes the output format required.
"""


class TextContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, context: str, model):
        super().__init__(model=model, user_query=user_query, context=context)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
            Evaluate the following metric:\n
            text_context_relevancy: Is the context provided by the text "{inputs["context"]}" relevant to the user\'s query "{inputs["user_query"]}"? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
            Give the reason as a string, not a list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        return [HumanMessage(content=[message])]


class ImageContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, image: str, model):
        super().__init__(model=model, user_query=user_query, image=[image])

    def get_prompt(self, inputs: dict):
        messages = []
        for image in inputs['image']:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                f"""
            Evaluate the following metric:\n
            image_context_relevancy: Is the context provided by the image(s) relevant to the user\'s query "{inputs["user_query"]}"? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
            Give the reason as a string, not a list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        messages.append(text_message)
        return [HumanMessage(content=messages)]


class AnswerRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, model):
        super().__init__(model=model, user_query=user_query, generated_answer=generated_answer)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
            Evaluate the following metric:\n
            answer_relevancy: Is the answer "{inputs["generated_answer"]}" relevant to the user\'s query "{inputs["user_query"]}"? (YES or NO)\n
            Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
            Give the reason as a string, not a list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        return [HumanMessage(content=[message])]


class AnswerCorrectnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, reference_answer: str, model):
        super().__init__(model=model, user_query=user_query, reference_answer=reference_answer,
                         generated_answer=generated_answer)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
                You are given a question, the correct reference answer, and the student\'s answer. \
                You are asked to grade the student\'s answer as either correct or incorrect, based on the reference answer. \
                Ignore differences in punctuation and phrasing between the student answer and true answer. \
                It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.\
                USER QUERY: "{inputs["user_query"]}"\n\
                REFERENCE ANSWER: "{inputs["reference_answer"]}"\n\
                STUDENT ANSWER: "{inputs["generated_answer"]}"\n\
                answer_correctness: Is the student's answer correct? (YES or NO)\n
                Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
                Give the reason as a string, not a list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        return [HumanMessage(content=[message])]


class ImageFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, image: str, model):
        super().__init__(user_query=user_query, generated_answer=generated_answer, image=[image], model=model)

    def get_prompt(self, inputs: dict):
        
        if not inputs["image"]:
            return None

            
        
        messages = []
        for image in inputs['image']:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                f"""
                Evaluate the following metric:\n
                image_faithfulness: Is the answer faithful to the context provided by the image(s), i.e. does it factually align with the context? (YES or NO)\n
                ANSWER: "{inputs["generated_answer"]}"\n\
                Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
                Give the reason as a string, not a list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        messages.append(text_message)
        return [HumanMessage(content=messages)]


class TextFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, context: str, model):
        super().__init__(user_query=user_query, generated_answer=generated_answer, context=context, model=model)

    def get_prompt(self, inputs: dict):
        
        if not inputs["context"]:
            return None
        
        message = {
            "type": "text",
            "text": (
                f"""
                Evaluate the following metric:\n
                text_faithfulness: Is the answer faithful to the context provided by the text, i.e. does it factually align with the context? (YES or NO)\n
                ANSWER: "{inputs["generated_answer"]}"\n\
                TEXT: "{inputs["context"]}"\n\
                Write out in a step by step manner your reasoning to be sure that your conclusion is correct.
                Give the reason as a string, not a list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        return [HumanMessage(content=[message])]
