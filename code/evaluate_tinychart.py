import os
import json
from natsort import natsorted
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
# from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds

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

def get_tinychart_response(folder_name, input_text, model_name, bool_list, question_index):
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

    if model_name != "TinyChart-3B-768":
        raise ValueError("Wrong model name")

    model_path = "mPLUG/TinyChart-3B-768"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device="cuda", # device="cpu" if running on cpu
    )

    image_file_names = natsorted([f for f in os.listdir(folder_name) if f.lower().endswith(('png', 'jpg', 'jpeg')) and f !="combined_image.png"])
    if bool_list[0] == 3:
        image_file_names = ["combined_image.png"]
    elif bool_list[0] == 2:
        input_text, image_index = specify_which_chart(input_text, folder_name)
        image_file_names = [image_file_names[image_index]]

    image_files = [os.path.join(folder_name, image_file_name) for image_file_name in image_file_names]
    # input_prompt = prefix[str(len(image_files))] + '\n' + input_text
    input_prompt = input_text

    response = inference_model(image_files, input_prompt, model, tokenizer, image_processor, context_len, conv_mode="phi",
                               max_new_tokens=1000)
    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)

