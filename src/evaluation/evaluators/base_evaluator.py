import os
from abc import abstractmethod
from langchain.chains.transform import TransformChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
#from langchain_openai import AzureChatOpenAI
import json
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.output_parsers import BooleanOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import BooleanOutputParser, OutputFixingParser
from transformers import LlavaNextForConditionalGeneration
from evaluation.evaluators.evaluator_interface import EvaluatorInterface
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import llava_call

class EvaluationResult(BaseModel):
    """The result of an evaluation for a given metric"""

    grade: str = Field(description="the grade after evaluating the metric (YES or NO)")
    reason: str = Field(description="The reasoning behind the grading decision")


class BaseEvaluator4o:
    """Centralized GPT-4o evaluator with metric-agnostic execution"""
    
    def __init__(self, client, model_name='gpt4o-240513', **kwargs):
        self.client = client
        self.model_name = model_name
        self.boolean_parser = BooleanOutputParser()
        self.kwargs = kwargs 

    def _validate_image(self, image_str):
        """Check basic structure of base64 image"""
        if not image_str.startswith("/9j/"):
            print("Warning: Image might have invalid header")
        if len(image_str) < 1000:
            print(f"Suspiciously short image string: {len(image_str)} chars")
        return image_str
        
    def run_evaluation(self):
        """Universal evaluation chain for all metrics"""
        chain = (
            RunnablePassthrough.assign(prompt=self.get_prompt)
            | RunnableLambda(self.call_gpt4o)
            | RunnableLambda(self.parse_response)
            | RunnableLambda(self._format_output)
        )
        return chain

    def _format_output(self, result):
        """Convert to BooleanOutputParser-compatible format"""
        #return {
         #   "grade": self.boolean_parser.parse(result["grade"]),
          #  "reason": result["reason"]
        #}
        #for 4o
        return result

    def call_gpt4o(self, inputs):
        """Execute GPT-4o API call with debug"""
        content = inputs['prompt']      
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {str(e)}")
            return {"grade": 0, "reason": "API Error"}

    def _format_content(self, prompt_data):
        """Format multimodal content for Azure API"""
        content = []
        
        if isinstance(prompt_data, list):
            for item in prompt_data:
                if item['type'] == 'text':
                    content.append({
                        "type": "text",
                        "text": item['text']
                    })
                elif item['type'] == 'image':
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{item['image']}",
                            "detail": "auto"
                        }
                    })
        else:  
            content.append({"type": "text", "text": prompt_data})
            
        return content

    def parse_response(self, response_text):
        """Extract JSON from GPT-4o response"""
        try:
            json_str = response_text.split("```json")[-1].replace("```", "").strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {"grade": "NO", "reason": "Invalid response format"}

    def _format_content(self, prompt_data):
        """Handle different input types including base64 images"""
        content = []
        
        if isinstance(prompt_data, list):
            for item in prompt_data:
                if item['type'] == 'text':
                    content.append({
                        "type": "text",
                        "text": item['text']
                    })
                elif item['type'] == 'image':
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": item['image_url'],
                            "detail": "auto"
                        }
                    })
        else:
            content.append({"type": "text", "text": str(prompt_data)})
            
        return content
    def get_prompt(self, inputs):
        """To be implemented by metric-specific evaluators"""
        raise NotImplementedError



class BaseEvaluator(EvaluatorInterface):
    """  
    A base class for an LLM evaluator.
  
    Attributes: 
        model (str): The model to be used for evaluation.
        tokenizer (LlavaNextProcessor or PreTrainedTokenizerFast): Tokenizer used for tokenization. Can be None.
        model_type (AzureChatOpenAI or LlavaNextForConditionalGeneration): Type of the model to use for evaluation.
        json_parser (JsonOutputParser): Parser used to parse evaluation results to a json object.
        boolean_parser (BooleanOutputParser): Parser used to parse the assigned grade to a boolean value.
        check_grade_chain (TransformChain): Applies the transformation from the LLM output for the grade to a boolean value.
        fix_format_parser (OutputFixingParser): Parser used to fix misformatted json output of an LLM.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        """  
        Initializes the BaseEvaluator object.
  
        :param model: The model to be used for evaluation.
        :param tokenizer: The tokenizer to be used for tokenization. Can be None.
        
        Keyword Args:
            user_query (str): The user query
            generated_answer (str): The answer produced by the model
            reference_answer (str): The ground truth answer
            context (str): The texts retrieved by the retrieval system
            image (str): The image retrieved by the retrieval system
        """
        self.model = model
        self.model_type = type(self.model)
        self.json_parser = JsonOutputParser(pydantic_object=EvaluationResult)
        self.boolean_parser = BooleanOutputParser()
        self.kwargs = kwargs
        self.check_grade_chain = TransformChain(
            input_variables=["grade", "reason"],
            output_variables=["grade", "reason"],
            transform=self.get_numeric_score
        )
        
        self.tokenizer = tokenizer
            
            
    def call_llava(self, inputs: dict) -> str:
        
        prompt = inputs['prompt']
        image = inputs.get('image', None)
        ans = llava_call(prompt, self.model, self.tokenizer, device="cuda", image=image)
        return ans
        

    def get_numeric_score(self, inputs: str) -> dict:
        """
        Checks that the obtained grade (YES or NO) can be parsed to a boolean and sets the grade to its integer value (0 ur 1)
        """
        inputs["grade"] = int(self.boolean_parser.parse(inputs["grade"]))
        return inputs

    def run_evaluation(self) -> dict:
        """  
        Performs evaluation for one output of a RAG system.
        Creates an evaluation chain that constructs the prompt, calls the model, fixes possible 
        json formatting errors and checks the validity of the assigned grade.

        :return: A json object with a grade (0 or 1) and a reason for the grading as string.
        """ 
        if self.tokenizer:
            chain = RunnableLambda(self.get_prompt) | RunnableLambda(self.call_llava)
            print("USING CALL LLAVA")
        else:
            print("NOT USING CALL LLAVA")
            chain = RunnableLambda(self.get_prompt) | self.model | self.json_parser | self.check_grade_chain
        result = chain.invoke(self.kwargs)

        return result

    @abstractmethod
    def get_prompt(self, inputs: dict):
        """
        Construct the prompt for evaluation based on a dictionary containing required input arguments.
        """
        pass
