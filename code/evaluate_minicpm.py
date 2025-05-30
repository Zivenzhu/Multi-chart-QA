from transformers import AutoModel, AutoTokenizer
import json
from PIL import Image
import torch
import os
from natsort import natsorted
from huggingface_hub import login
login(token="") # Replace with your actual Hugging Face token


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


def get_minicpm_response(folder_name, input_text, model_name, bool_list, question_index):
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

    # sdpa attn impl for v2.6, default for 2 and 2.5
    if model_name == "MiniCPM-V-2_6":
        model_path = "openbmb/MiniCPM-V-2_6"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    elif model_name == "MiniCPM-Llama3-V-2_5":
        model_path = "openbmb/MiniCPM-Llama3-V-2_5"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        raise NotImplementedError
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    msgs = []
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

    msgs.extend([dict(type='image', value=p) for p in image_file_names])
    msgs.append(dict(type='text', value=input_text))
    content = []
    for x in msgs:
        if x['type'] == 'text':
            content.append(x['value'])
        elif x['type'] == 'image':
            img = Image.open(x['value'])
            max_size = 512
            if max(img.size) > max_size:
                scale = max_size / max(img.size)
                new_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            image = img.convert('RGB')
            content.append(image)
    msgs = [{'role': 'user', 'content': content}]
    # try:
    response = model.chat(
        msgs=msgs,
        context=None,
        image=None,
        tokenizer=tokenizer,
        sampling=False,
        temperature=0.0,
        top_p=1.0,
    )
    # except:
    #     response = 'Error occur when generating response.'

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)