import base64
from evaluation.evaluators.base_evaluator import BaseEvaluator4o
from typing import Dict, Any

class AnswerCorrectnessEvaluator(BaseEvaluator4o):
    """Evaluates if generated answer matches reference answer"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.generated_answer = kwargs.get('generated_answer')
        self.reference_answer = kwargs.get('reference_answer')

    def get_prompt(self, inputs: Dict[str, Any]) -> list:
        return [
            {
                "type": "text",
                "text": f"""**Task:** Evaluate answer correctness (1-5 scale)
1 = Completely incorrect
5 = Perfect match

**Question:** {self.user_query}
**Reference Answer:** {self.reference_answer}
**Generated Answer:** {self.generated_answer}

**Response Format:** 
{{
  "grade": "1-5", 
  "reason": "detailed explanation"
}}"""
            }
        ]

class AnswerRelevancyEvaluator(BaseEvaluator4o):
    """Evaluates if answer directly addresses the question"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.generated_answer = kwargs.get('generated_answer')

    def get_prompt(self, inputs: Dict[str, Any]) -> list:
        return [
            {
                "type": "text",
                "text": f"""**Task:** Evaluate answer relevancy (1-5 scale)
1 = Completely irrelevant
5 = Fully addresses question

**Question:** {self.user_query}
**Generated Answer:** {self.generated_answer}

**Response Format:** 
{{
  "grade": "1-5", 
  "reason": "detailed explanation"
}}"""
            }
        ]

class ImageFaithfulnessEvaluator(BaseEvaluator4o):
    """Evaluates if image content matches generated answer"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.generated_answer = kwargs.get('generated_answer')
        self.image = kwargs.get('image')



    def get_prompt(self, inputs: Dict[str, Any]) -> list:
        print(f"Building prompt with image size: {len(self.image)} chars")

        binary_data = base64.b64decode(self.image)
        ascii_encoded_data = base64.b64encode(binary_data).decode('ascii')
        return [
            {
                "type": "text",
                "text":  f"""**Task:** Evaluate image faithfulness of the generated answer compared to the provided image (1-5 scale) 
1 = Contradicts image content
5 = Perfectly describes image

**Question:** {self.user_query}
**Generated Answer:** {self.generated_answer}"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ascii_encoded_data}",
                    "detail": "auto"
                }
            },
            {
                "type": "text",
                "text": """       "text": "Analyze THIS IMAGE to verify the answer:"**Response Format:** 
{
  "grade": "1-5", 
  "reason": "detailed explanation"
}"""
            }
        ]


class TextFaithfulnessEvaluator(BaseEvaluator4o):
    """Evaluates if text answer stays grounded in provided context"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.generated_answer = kwargs.get('generated_answer')
        self.context = kwargs.get('context')

    def get_prompt(self, inputs: Dict[str, Any]) -> list:
        return [
            {
                "type": "text",
                "text": f"""**Task:** Evaluate text faithfulness (1-5 scale)
1 = Contains significant hallucinations
5 = Fully supported by context

**Question:** {self.user_query}
**Context:** {self.context}
**Generated Answer:** {self.generated_answer}

**Response Format:** 
{{
  "grade": "1-5", 
  "reason": "detailed explanation"
}}"""
            }
        ]

class ImageContextRelevancyEvaluator(BaseEvaluator4o):
    """Evaluates if retrieved image is relevant to question"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.image = kwargs.get('image')

    def get_prompt(self, inputs: Dict[str, Any]) -> list:

        binary_data = base64.b64decode(self.image)
        ascii_encoded_data = base64.b64encode(binary_data).decode('ascii')


        return [
            {
                "type": "text",
                "text": f"""**Task:** Evaluate image relevancy of the provided image to the given question (1-5 scale)
1 = Completely irrelevant
5 = Perfectly matches question

**Question:** {self.user_query}"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ascii_encoded_data}",
                    "detail": "auto"
                }
            },
            {
                "type": "text",
                "text": """**Response Format:** 
{
  "grade": "1-5", 
  "reason": "detailed explanation"
}"""
            }
        ]

class TextContextRelevancyEvaluator(BaseEvaluator4o):
    """Evaluates if retrieved text is relevant to question"""
    
    def __init__(self, client, model_name, **kwargs):
        super().__init__(client, model_name, **kwargs)
        self.user_query = kwargs.get('user_query')
        self.context = kwargs.get('context')

    def get_prompt(self, inputs: Dict[str, Any]) -> list:
        return [
            {
                "type": "text",
                "text": f"""**Task:** Evaluate text relevancy (1-5 scale)
1 = Completely irrelevant
5 = Directly answers question

**Question:** {self.user_query}
**Retrieved Text:** {self.context}

**Response Format:** 
{{
  "grade": "1-5", 
  "reason": "detailed explanation"
}}"""
            }
        ]