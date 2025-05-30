import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import os
from natsort import natsorted
import json


def specify_which_chart(input_text, folder_name):
    if "the first chart" in input_text:
        image_index = 0
        input_text = input_text.replace("the first chart", "the chart")
    elif "the second chart" in input_text:
        image_index = 1
        input_text = input_text.replace("the second chart", "the chart")
    elif "the third chart" in input_text:
        image_index = 2
        input_text = input_text.replace("the third chart", "the chart")
    else:
        print(folder_name)
        print(input_text)
        raise NotImplementedError("Cannot specify which chart!")
    return input_text, image_index

def open_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    if width < height:
        scale = input_size / width
    else:
        scale = input_size / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height))
    return image

def get_llava_ov_response(folder_name, input_text, model_name, bool_list, question_index):
    tmp_folder_list = [["with_chart_reference", "without_chart_reference", "only_single_image_input", "merged_image"], ["with_CoT", "without_CoT", "text_only"]]

    folder_path_with_prefix = ''
    for i in range(len(bool_list)):
        tmp_folder_name = tmp_folder_list[i][bool_list[i]]
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, tmp_folder_name)
        os.makedirs(folder_path_with_prefix, exist_ok=True)

    parts = folder_name.split('/')
    folder_type = f'evaluation_{model_name}'
    group_folder_name = parts[-1]
    os.makedirs(os.path.join(folder_path_with_prefix, folder_type), exist_ok=True)
    group_folder_path = os.path.join(folder_path_with_prefix, folder_type, group_folder_name)
    os.makedirs(group_folder_path, exist_ok=True)

    file_path = os.path.join(group_folder_path, f'output_of_question_{question_index}.json')
    if os.path.exists(file_path):
        return

    if model_name == "llava-onevision-qwen2-7b-ov-hf":
        model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    elif model_name == "llava-onevision-qwen2-72b-ov-hf":
        model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
    else:
        raise ValueError("Wrong model name")

    # Load the model with  device_map
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        device_map="auto"  # Automatically assign the model to the available devices
    )

    processor = AutoProcessor.from_pretrained(model_id)

    image_file_names = natsorted(
        [f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"],
        key=lambda x: x.lower()
    )
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]

    image_file_names = [os.path.join(folder_name, image_file_name) for image_file_name in image_file_names]

    if bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]
    image_files = [open_image(os.path.join(folder_name, image_file_name)) for image_file_name in image_file_names]

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text}
                ] +
                [
                    {"type": "image"} for _ in range(len(image_files))
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image_files, text=prompt, return_tensors="pt").to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    response = processor.decode(output[0][2:], skip_special_tokens=True)
    prompt_prefix = "assistant"
    response = response.split(prompt_prefix)[1].strip()

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)
