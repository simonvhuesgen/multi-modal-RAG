import os
import requests

# Configuration
GPT4V_KEY = os.environ.get("GPT4V_API_KEY")
GPT4V_ENDPOINT = os.environ.get("GPT4V_ENDPOINT")


def gpt4v_qa_prompt_template(system_prompt, question, context, image=None, temperature=0.7, top_p=0.95, max_tokens=300):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{context}\n\nQuestion:\n{question}\n"
                    }
                ]
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    if image:
        img_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}"
            }
        }
        payload["messages"][1]["content"].append(img_dict)

    return payload


def gpt4v_dataset_generation_prompt_template(system_prompt, text, image=None, temperature=0.7, top_p=0.95,
                                             max_tokens=300):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{text}\n\nImage:\n"
                    }
                ]
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    if image:
        img_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}"
            }
        }
        payload["messages"][1]["content"].append(img_dict)

    return payload


def gpt4v_call(system_prompt, text, task, question=None, image=None, temperature=0.7, top_p=0.95, max_tokens=300):
    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    if task == "qa":
        json_prompt = gpt4v_qa_prompt_template(system_prompt, question, text, image, temperature, top_p, max_tokens)
    else:   # synthetic dataset generation
        json_prompt = gpt4v_dataset_generation_prompt_template(system_prompt, text, image, temperature, top_p, max_tokens)

    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=json_prompt)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response.json()["choices"][0]["message"]["content"]
