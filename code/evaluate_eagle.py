import os
import torch
import json
from natsort import natsorted
from PIL import Image

#Please download the eagle repository from https://github.com/NVlabs/EAGLE/tree/main/Eagle1/eagle
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria



Image.MAX_IMAGE_PIXELS = None
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

# prefix = {
#     "1": "<image>. ",
#     "2": "The first chart is <image>. \n The second chart is <image>. ",
#     "3": "The first chart is <image>. \n The second chart is <image>. \n The third chart is <image>. ",
# }

def get_eagle_response(folder_name, input_text, model_name, bool_list, question_index):
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

    if model_name != "Eagle-X5-13B-Chat":
        raise ValueError("Wrong model name")

    model_path = "NVEagle/Eagle-X5-13B-Chat"
    conv_mode = "vicuna_v1"
    input_prompt = input_text

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,None,get_model_name_from_path(model_path), device_map="cuda:0")
    if model.config.mm_use_im_start_end:
        input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
    else:
        input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_file_names = natsorted([f for f in os.listdir(folder_name) if
                                  f.lower().endswith(('png', 'jpg', 'jpeg')) and f != "combined_image.png"])
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]
    image_files = [Image.open((os.path.join(folder_name, image_file_name))).convert('RGB') for image_file_name in image_file_names]

    image_tensors = process_images(image_files, image_processor, model.config)
    image_tensors = [image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True) for image_tensor in image_tensors]
    image_sizes = [image.size for image in image_files]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.to(device='cuda:0', non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=1000,
            use_cache=True)

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)
