from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
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

def get_llava_response(folder_name, question, model_name, bool_list, question_index):
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

    # Load the model in half-precision
    model_path = os.path.join("llava-hf", model_name)
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()

    # Get three different images
    image_file_names = natsorted(
        [f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"],
        key=lambda x: x.lower()
    )
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        question, image_index = specify_which_chart(question, folder_name)
        image_file_names = [image_file_names[image_index]]
    image_files = [Image.open(os.path.join(folder_name, image_file_name)).convert('RGB').resize((448, 448)) for image_file_name in image_file_names]

    if 'llava-v1.6-mistral-7b-hf' in model_path:
        prompt_prefix = "[/INST]"
        max_token = 1000
        if len(image_files) == 2:
            prompt = f"[INST] <image> \n<image> \n{question} [/INST]"
        elif len(image_files) == 3:
            prompt = f"[INST] <image> \n<image> \n<image> \n{question} [/INST]"
        elif len(image_files) == 1:
            prompt = f"[INST] <image> \n{question} [/INST]"
        else:
            raise ValueError("Image amount should be 1 or 2 or 3")
    elif 'llava-v1.6-vicuna-13b-hf' in model_path:
        prompt_prefix = "ASSISTANT:"
        max_token = 1000
        if len(image_files) == 2:
            prompt = f"USER: <image>\n<image>\n{question} ASSISTANT:"
        elif len(image_files) == 3:
            prompt = f"USER: <image>\n<image>\n<image>\n{question} ASSISTANT:"
        elif len(image_files) == 1:
            prompt = f"USER: <image>\n{question} ASSISTANT:"
        else:
            raise ValueError("Image amount should be 1 or 2 or 3")
    elif 'llava-v1.6-34b-hf' in model_path:
        prompt_prefix = "<|im_start|> assistant"
        max_token = 1000
        if len(image_files) == 2:
            prompt = f"<|im_start|>system\nAnswer the question.<|im_end|><|im_start|>user\n<image>\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
        elif len(image_files) == 3:
            prompt = f"<|im_start|>system\nAnswer the question.<|im_end|><|im_start|>user\n<image>\n<image>\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
        elif len(image_files) == 1:
            prompt = f"<|im_start|>system\nAnswer the question.<|im_end|><|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
        else:
            raise ValueError("Image amount should be 1 or 2 or 3")
    else:
        raise ValueError("Wrong model name")


    # We can simply feed images in the order they have to be used in the text prompt
    # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
    inputs = processor(text=prompt, images=image_files, return_tensors="pt").to("cuda:0")

    # Generate
    output = model.generate(**inputs, max_new_tokens=max_token, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)
    response = response.split(prompt_prefix)[1].strip()

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)