import math
import numpy as np
import torch
import torchvision.transforms as T
import os
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from natsort import natsorted

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    if width < height:
        scale = input_size / width
    else:
        scale = input_size / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height))

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

# def split_model(model_name):
#     device_map = {}
#     world_size = torch.cuda.device_count()
#     num_layers = {
#         'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
#         'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
#     # Since the first GPU will be used for ViT, treat it as half a GPU.
#     num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
#     num_layers_per_gpu = [num_layers_per_gpu] * world_size
#     num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
#     layer_cnt = 0
#     for i, num_layer in enumerate(num_layers_per_gpu):
#         for j in range(num_layer):
#             device_map[f'language_model.model.layers.{layer_cnt}'] = i
#             layer_cnt += 1
#     device_map['vision_model'] = 0
#     device_map['mlp1'] = 0
#     device_map['language_model.model.tok_embeddings'] = 0
#     device_map['language_model.model.embed_tokens'] = 0
#     device_map['language_model.output'] = 0
#     device_map['language_model.model.norm'] = 0
#     device_map['language_model.lm_head'] = 0
#     device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
#
#     return device_map

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_internvl15_response(folder_name, input_text, model_name, bool_list, question_index):
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

    if model_name == "InternVL-Chat-V1-5":
        model_path = 'OpenGVLab/InternVL-Chat-V1-5'
        device_map = "auto"
        # device_map = split_model('InternVL-Chat-V1-5')
    elif model_name == "InternVL2-26B":
        model_path = "OpenGVLab/InternVL2-26B"
        # device_map = split_model(model_name)
        device_map = "auto"
    elif model_name == "InternVL2-Llama3-76B":
        model_path = "OpenGVLab/InternVL2-Llama3-76B"
        # device_map = split_model(model_name)
        device_map = "auto"
    else:
        raise ValueError("Wrong model path")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    generation_config = dict(
        num_beams=1,
        max_new_tokens=1000,
        do_sample=False,
    )

    image_file_names = natsorted(
        [f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"],
        key=lambda x: x.lower()
    )
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]

    pixel_value_list = [load_image(image_file=os.path.join(folder_name, image_file_name), max_num=9).to(torch.bfloat16).cuda() for image_file_name
                        in image_file_names]
    pixel_values = torch.cat(pixel_value_list, dim=0)

    prefix = {
        "1": "<image>. ",
        "2": "The first chart is <image>. The second chart is <image>. ",
        "3": "The first chart is <image>. The second chart is <image>. The third chart is <image>. ",
    }

    if model_name == "InternVL-Chat-V1-5":
        question = prefix[str(len(image_file_names))] + '\n' + input_text  ## intervl_1.5
    elif model_name == "InternVL2-26B":
        question = f"<image>\n{input_text}"  ## intervl2_26B
    else:
        raise ValueError("Wrong value for question")

    if model_name == "InternVL-Chat-V1-5":
        num_patches_list = [pixel_value.size(0) for pixel_value in pixel_value_list]
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)
    else:

        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       history=None, return_history=True)

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)
