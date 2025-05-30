import base64
import requests
import os
from natsort import natsorted
import json
import time

#gpt4-para
api_key= "" # Replace with your actual API key
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def process_images_in_folder(folder_name, bool_list):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_imgs = []
    if bool_list[0] == 3:
        image_files = ["combined_image.png"]
    else:
        image_files = natsorted(
            [f for f in os.listdir(folder_name) if
             f.lower().endswith(('png', 'jpg', 'jpeg')) and f != "combined_image.png"],
            key=lambda x: x.lower()
        )
    for name in image_files:
        file_path = os.path.join(folder_name, name)
        base64_imgs.append(encode_image(file_path))
    return base64_imgs

def generate_gpt_response(folder_name, input_text, model_name, bool_list, question_index):
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

    if bool_list[1] == 2:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text
                        },
                    ]}
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_data = response.json()
            with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
                json.dump(response_data, outfile, indent=4)
            time.sleep(4)
        except:
            time.sleep(4)
        return

    base64_imgs = process_images_in_folder(folder_name, bool_list)
    if bool_list[0] != 2:
        payload = {
            "model": model_name,
            "messages": [
            {
                "role": "user",
                 "content": [
                     {
                        "type": "text",
                        "text": input_text
                     }] +
                    [{
                        "type": "image_url",
                        "image_url": {
                             "url": f"data:image/jpeg;base64,{base64_imgs[i]}"
                        }
                     } for i in range(len(base64_imgs))
                 ]}
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }
    else:
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
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_imgs[image_index]}"
                            }
                        },
                    ]}
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42
        }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
            json.dump(response_data, outfile, indent=4)
        time.sleep(4)
    except:
        time.sleep(4)
