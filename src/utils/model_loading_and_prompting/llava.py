from transformers import BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import io
import pandas as pd
from typing import Tuple
from rag_env import INPUT_DATA


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attention_2=True
)

def format_prompt_with_image(prompt: str) -> str:
    return f"[INST] <image>\n{prompt} [/INST]"


def get_qa_prompt(model_id:str, system_prompt:str, question: str, context: str, image: Image=None) -> str:
    if "vicuna" in model_id:
        prompt = f"USER:{'<image>' if image else ' '}\n{system_prompt}\n{context}\n\nQuestion:\n{question}\n\nASSISTANT:"
    else:   # mistral
        prompt = f"[INST]{'<image>' if image else ' '}\n{system_prompt}\n{context}\n\nQuestion:\n{question}\n\n[/INST]"

    return prompt


def get_dataset_generation_prompt(model_id: str, system_prompt: str, context: str, image: Image=None):
    if "vicuna" in model_id:
        prompt = f"USER:{'<image>' if image else ' '}\n{system_prompt}\n{context}\n\nASSISTANT:"
    else:   # mistral
        prompt = f"[INST]{'<image>' if image else ' '}\n{system_prompt}\n{context}\n\n[/INST]"

    return prompt


def format_output(raw_output, processor: LlavaNextProcessor, prompt: str) -> str:
    out = processor.decode(raw_output[0], skip_special_tokens=True)
    out_prompt = prompt.replace("<image>", " ").strip()
    formatted_output = out.replace(out_prompt, "").strip()

    return formatted_output


def get_prompt(task: str, model_id: str, system_prompt: str, text: str, image: Image, question: str) -> str:
    if task == "qa":
        prompt = get_qa_prompt(model_id, system_prompt, question, text, image)
    else:
        prompt = get_dataset_generation_prompt(model_id, system_prompt, text, image)
    return prompt


def llava_call(prompt: str, model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor, device: str, image: Image=None) -> str:

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    raw_output = model.generate(**inputs, max_new_tokens=300)
    formatted_output = format_output(raw_output, processor, prompt)

    return formatted_output


def load_llava_model(model_id: str) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    processor = LlavaNextProcessor.from_pretrained(model_id)
    # uncomment to use quantized version of the model
    # model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    return model, processor


if __name__ == "__main__":
    index = 56
    device = "cuda"
    model, processor = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")
    model = model.eval()
    
    df = pd.read_parquet(INPUT_DATA)

    qa_system_prompt = "You are an AI assistant that answers question from the industrial domain based on a given context. Use the information from the context to answer the question. \nContext:\n"
    qa_system_prompt_img = "You are an AI assistant that answers question from the industrial domain based on a given context as text and image. Use both the information from text and image to answer the question. \nContext:\n"

    question = "What are the possible positions of the manual operator and what colors are associated with each position?"
    context = df["text"][index]
    img_bytes = df["image_bytes"][index]
    image = Image.open(io.BytesIO(img_bytes))
    
    img_prompt = get_qa_prompt("llava-hf/llava-v1.6-mistral-7b-hf", qa_system_prompt_img, question, context, image)
    no_img_prompt = get_qa_prompt("llava-hf/llava-v1.6-mistral-7b-hf", qa_system_prompt, question, context)

    # get response from image and text context
    print("============== Answer with image:")
    llava_response_img = llava_call(img_prompt, model, processor, device, image)
    print(llava_response_img)
    
    # get response using text only
    print("============== Answer without image:")
    llava_response_no_img = llava_call(no_img_prompt, model, processor, device)
    print(llava_response_no_img)
