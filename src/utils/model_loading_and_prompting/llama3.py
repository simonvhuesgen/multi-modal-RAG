from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
import torch

# demo code for summarizing text using LLama3 8B Instruct from HuggingFace

text = """Language model pre-training has been shown to
be effective for improving many natural language
processing tasks (Dai and Le, 2015; Peters et al.,
2018a; Radford et al., 2018; Howard and Ruder,
2018). These include sentence-level tasks such as
natural language inference (Bowman et al., 2015;
Williams et al., 2018) and paraphrasing (Dolan
and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them
holistically, as well as token-level tasks such as
named entity recognition and question answering,
where models are required to produce fine-grained
output at the token level (Tjong Kim Sang and
De Meulder, 2003; Rajpurkar et al., 2016).
There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The
feature-based approach, such as ELMo (Peters
et al., 2018a), uses task-specific architectures that
include the pre-trained representations as additional features. The fine-tuning approach, such as
the Generative Pre-trained Transformer (OpenAI
GPT) (Radford et al., 2018), introduces minimal
task-specific parameters, and is trained on the
downstream tasks by simply fine-tuning all pretrained parameters. The two approaches share the
same objective function during pre-training, where
they use unidirectional language models to learn
general language representations.
"""


template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_msg}
                <|start_header_id|>user<|end_header_id|>Text: {text}\nSummary:\n<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>"""

system_msg = """You are an assistant tasked with summarizing text for retrieval.
            These summaries will be embedded and used to retrieve the raw text elements.
            Give a concise summary of the text that is well optimized for retrieval.
            Only output the summary, no additional explanation.\n"""
            

prompt = PromptTemplate.from_template(template)

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}


pipe = pipeline(
"text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", **generation_params
)

model = HuggingFacePipeline(pipeline=pipe)


summarize_chain = prompt | model
res = summarize_chain.invoke({"system_msg": system_msg, "text": text})
output = res.split("<|start_header_id|>assistant<|end_header_id|>",1)[1].strip()
print("Summary: ", output)