import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from natsort import natsorted
import json
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def resize_image(img):
    max_size = 1024
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    # img = img.convert('RGB')
    return img


api_key = "" # Replace with your actual API key
def get_gemini_response(folder_name, input_text, model_name, bool_list, question_index):
    # genai.configure(api_key=os.environ["API_KEY"])
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    image_file_names = natsorted(
        [f for f in os.listdir(folder_name) if
         f.lower().endswith(('png', 'jpg', 'jpeg')) and f != "combined_image.png"],
        key=lambda x: x.lower()
    )

    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    image_files = [resize_image(Image.open(os.path.join(folder_name, image_file_name))) for image_file_name in image_file_names]

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

    if bool_list[0] == 2:
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
        image_files = [image_files[image_index]]

    contents = [input_text] + image_files

    try:
        response = model.generate_content(
            contents, stream=True,
            generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=1000,
            temperature=0.0,
            top_p=1.0,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
        response.resolve()
        response = response.text
        with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
            json.dump(response, outfile, indent=4)
        time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Folder name: {folder_name}")
        exit(0)

