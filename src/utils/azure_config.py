def get_azure_config():
    return {
        'gpt4': {
            'openai_endpoint': 'YOUR_ENDPOINT_HERE',
            'deployment_name': 'gpt-4',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4',
        },
        'gpt4_vision': {
            'openai_endpoint': 'YOUR_ENDPOINT_HERE',
            'deployment_name': 'gpt-4-vision-preview',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4-vision-preview',
        },
        'text_embedding_3': {
            'openai_endpoint': 'YOUR_ENDPOINT_HERE',
            'deployment_name': 'text-embedding-3-small',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'text-embedding-3-small',
        },
        'gpt3.5': {
            'openai_endpoint': 'YOUR_ENDPOINT_HERE',
            'deployment_name': 'gpt-35-turbo',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-35-turbo',
        },
        'gpt4o': {
            'openai_endpoint': 'YOUR_ENDPOINT_HERE',
            'deployment_name': 'gpt4o',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt4o-240513',
            'api_key': 'YOUR_KEY_HERE',
        },
    }