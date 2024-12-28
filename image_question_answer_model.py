# image_question_answer_model.py
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model():
    """
    Load the pre-trained Llava model and processor
    Returns:
        tuple: (model, processor)
    """
    # You can try 13b model based on your GPU processing capacity may or may work.
    #llava-hf/llava-1.5-13b-hf
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return model, processor

def generate_response(model, processor, image, question):
    """
    Generate a response for the given image and question
    Args:
        model: The LLaVA model
        processor: The model processor
        image: PIL Image object
        question: str, the question about the image
    Returns:
        str: The model's response
    """
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=100)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response