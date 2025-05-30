import torch
from PIL import Image
import os
from natsort import natsorted
from transformers import AutoProcessor, AutoModelForVision2Seq
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

def process_image(img):
    max_size = 512
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    img = img.convert('RGB')
    return img

DEVICE = "cuda"
def get_idefics_response(folder_name, input_text, model_name, bool_list, question_index):
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

    image_file_names = natsorted(
        [f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"],
        key=lambda x: x.lower()
    )

    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]
    image_files = [process_image(Image.open(os.path.join(folder_name, image_file_name))) for image_file_name in image_file_names]

    if model_name == 'idefics2-8b':
        model_path = "HuggingFaceM4/idefics2-8b"
        prompt_prefix = "Assistant:"
    elif model_name == 'Idefics3-8B-Llama3':
        model_path = "HuggingFaceM4/Idefics3-8B-Llama3"
        prompt_prefix = "Assistant:"
    else:
        raise NotImplementedError

    processor = AutoProcessor.from_pretrained(model_path, device_map="auto")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if len(image_files) == 2:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image"},
                {"type": "image"},
            ],
        }]
    elif len(image_files) == 3:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
            ],
        }]
    elif len(image_files) == 1:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image"},
            ],
        }]
    else:
        raise NotImplementedError
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image_files, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_texts.split(prompt_prefix)[1].strip()

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)
