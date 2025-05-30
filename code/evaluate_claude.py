import anthropic
import base64
import json
import os
from natsort import natsorted
import time
import io
from PIL import Image

api_key = "" # Replace with your actual API key
def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Claude"
    assert model_path is not None, "Model name is required for using Claude"
    client = anthropic.Anthropic(api_key=api_key)
    model = model_path
    return client, model

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

def process_images_in_folder(folder_name, bool_list, question):
    def encode_image(image_path):
        with Image.open(image_path) as img:
            max_size = 2048
            if max(img.size) > max_size:
                scale = max_size / max(img.size)
                new_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    base64_imgs = []
    image_files = natsorted(
        [f for f in os.listdir(folder_name) if
         f.lower().endswith(('png', 'jpg', 'jpeg')) and f != "combined_image.png"],
        key=lambda x: x.lower()
    )
    if bool_list[0] == 2:
        question, image_index = specify_which_chart(question, folder_name)
        image_files = [image_files[image_index]]
    elif bool_list[0] == 3:
        image_files = ["combined_image.png"]

    for name in image_files:
        file_path = os.path.join(folder_name, name)
        base64_imgs.append(encode_image(file_path))
    return base64_imgs, question

def get_claude_response(folder_name, input_text, model_name, bool_list, question_index):
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

    try:
        client = anthropic.Anthropic(api_key=api_key)
        image_base64_coded_list, input_text = process_images_in_folder(folder_name, bool_list, input_text)
        media_type = 'image/png'
        message = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64_coded,
                            },
                        } for image_base64_coded in image_base64_coded_list] +
                    [
                        {
                            "type": "text",
                            "text": input_text,
                        }
                    ],
                }
            ],
        )
        message = message.json()
        message = json.loads(message)
        response = message['content'][0]['text']
        with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
            json.dump(response, outfile, indent=4)
        time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Folder name: {folder_name}")
        exit(0)
