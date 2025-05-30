from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from natsort import natsorted
import os
import json

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

prefix = {
    "1": "<image>. ",
    "2": "The first chart is <image>. \n The second chart is <image>. ",
    "3": "The first chart is <image>. \n The second chart is <image>. \n The third chart is <image>. ",
}

def generate_chartgemma_response(folder_name, input_text, model_name, bool_list, question_index):
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

    if model_name != 'chartgemma':
        raise ValueError("Wrong model name")

    model_path = "ahmed-masry/chartgemma"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_file_names = natsorted([f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"])
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]

    image_files = [Image.open((os.path.join(folder_name, image_file_name))).convert('RGB') for image_file_name in image_file_names]
    input_prompt = prefix[str(len(image_files))] + '\n' + input_text
    inputs = processor(text=input_prompt, images=image_files, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # change type if pixel_values in inputs to fp16.
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    prompt_length = inputs['input_ids'].shape[1]

    # Generate
    generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=1000)
    response = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)