import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
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


def get_deepseek_response(folder_name, input_text, model_name, bool_list, question_index):
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

    # file_path = os.path.join(group_folder_path, f'output_of_question_{question_index}.json')
    # if os.path.exists(file_path):
    #     return

    # specify the path to the model
    if model_name == "deepseek-vl-7b-chat":
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
    else:
        raise ValueError("Not correct model name")

    # Load the VLChatProcessor 
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the MultiModalityCausalLM model 
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
                                                                         trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

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

    prefix = {
        "1": "<image_placeholder>. ",
        "2": "The first chart is <image_placeholder>. The second chart is <image_placeholder>. ",
        "3": "The first chart is <image_placeholder>. The second chart is <image_placeholder>. The third chart is <image_placeholder>. ",
    }

    conversation = [
        {
            "role": "User",
            "content": prefix[str(len(image_file_names))] + '\n' + input_text,
            "images": image_file_names
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1000,
        do_sample=False,
        use_cache=True
    )

    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    with open(os.path.join(group_folder_path, f'output_of_question_{question_index}.json'), 'w') as outfile:
        json.dump(response, outfile, indent=4)
