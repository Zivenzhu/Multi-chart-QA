import os
import chardet
import json
from evaluate_gpt import generate_gpt_response
# from evaluate_llava import get_llava_response
# from evaluate_idefics import get_idefics_response
# from evaluate_minicpm import get_minicpm_response
# from evaluate_deepseek import get_deepseek_response
# from evaluate_internvl15 import get_internvl15_response
# from evaluate_claude import get_claude_response
# from evaluate_gemini import get_gemini_response
# from evaluate_qwen_local import generate_qwen_local_response
# from evaluate_llava_ov import get_llava_ov_response
# from evaluate_chartgemma import generate_chartgemma_response
# from evaluate_matcha import get_matcha_response
# from evaluate_tinychart import get_tinychart_response
# from evaluate_eagle import get_eagle_response

head_with_CoT = """Here is a question for you to solve.
You should give the response in the following format.
"Solution: (Here you are allowed to use chain of thought and provide short explanations or intermediate calculation steps.)
Answer: (You must output the answer in the format specified in the question without giving any explanations or intermediate calculation steps.)"

"""

head_without_CoT = """Here is a question for you to solve.
You should avoid providing any explanations in your response and only output the answer of this question in the format as follows.
"Answer: (You must output the answer in the format specified in the question without giving any explanations or intermediate calculation steps.)"

"""

head_without_charts = """You should try to give a reasonable answer based on the question only. 
You should give the response in the following format.
"Solution: (Here you are allowed to use chain of thought and provide short explanations or intermediate calculation steps.)
Answer: (You must output the answer in the format specified in the question without giving any explanations or intermediate calculation steps.)"

"""

head_random = """
Randomly guess a reasonable answer based on the question only.
If the question asks for a number, you can randomly guess a number within a reasonable range.
If the question asks for a term, you can randomly guess a term that is relevant to the question.
You should give the response in the following format.
"Solution: (Here you are allowed to use chain of thought and provide short explanations or intermediate calculation steps.)
Answer: (You must output the answer in the format specified in the question without giving any explanations or intermediate calculation steps.)"

"""


def open_json_file(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data, encoding


bool_lists = [[0, 0]]
# Please set bool_lists to [[0, 0]] as the default configuration for basic evaluation.
for bool_list in bool_lists:
    # (with_chart_reference, without_chart_reference, single_image, merged_image) (with_CoT, without_CoT, text_only)

    base_folder = "../data"
    group_list = os.listdir(base_folder)
    for group_folder in group_list:
            group_path = os.path.join(base_folder, group_folder)
            qa_json_file_name = os.path.join(group_path, 'chart-path_and_question-answer_pair.json')
            qa_json_files, json_encoding = open_json_file(qa_json_file_name)

            if bool_list[1] == 0:
                head = head_with_CoT
            elif bool_list[1] == 1:
                head = head_without_CoT
            elif bool_list[1] == 2:
                head = head_without_charts
            else:
                raise ValueError("The first element in bool list should be 0-2")

            for i in range(4):
                # if bool_list[0] == 1:
                #     if "required" not in qa_json_files[i]["task"]:
                #         continue
                # elif bool_list[0] == 2:
                #     if "Direct_Questions" not in qa_json_files[i]["task"]:
                #         continue
                question_string_tmp = qa_json_files[i]["question"]
                all_question_tmp = head + question_string_tmp
                generate_gpt_response(group_path, all_question_tmp, 'gpt-4o', bool_list, i+1)
                # generate_gpt_response(group_path, all_question_tmp, 'gpt-4o-mini', bool_list, i+1)
                # get_llava_response(group_path, all_question_tmp, 'llava-v1.6-mistral-7b-hf', bool_list, i+1)
                # get_idefics_response(group_path, all_question_tmp, 'idefics2-8b', bool_list, i+1)
                # get_minicpm_response(group_path, all_question_tmp, "MiniCPM-Llama3-V-2_5", bool_list, i+1)
                # get_deepseek_response(group_path, all_question_tmp, "deepseek-vl-7b-chat", bool_list, i+1)
                # get_internvl15_response(group_path, all_question_tmp, "InternVL-Chat-V1-5", bool_list, i+1)


                # generate_qwen_local_response(group_path, all_question_tmp, "Qwen2-VL-7B-Instruct", bool_list, i+1)
                # get_idefics_response(group_path, all_question_tmp, 'Idefics3-8B-Llama3', bool_list, i + 1)
                # get_llava_ov_response(group_path, all_question_tmp, "llava-onevision-qwen2-7b-ov-hf", bool_list, i + 1)
                # get_claude_response(group_path, all_question_tmp, "claude-3-5-sonnet-20240620", bool_list, i+1)
                # get_gemini_response(group_path, all_question_tmp, "gemini-1.5-pro", bool_list, i+1)


                # get_internvl15_response(group_path, all_question_tmp, "InternVL2-26B", bool_list, i + 1)
                # get_minicpm_response(group_path, all_question_tmp, "MiniCPM-V-2_6", bool_list, i + 1)
                # generate_chartgemma_response(group_path, all_question_tmp, "chartgemma", bool_list, i + 1)
                # get_matcha_response(group_path, all_question_tmp, "matcha-chartqa", bool_list, i + 1)
                # get_tinychart_response(group_path, all_question_tmp, "TinyChart-3B-768", bool_list, i + 1)
                # get_eagle_response(group_path, all_question_tmp, "Eagle-X5-13B-Chat", bool_list, i + 1)


                # generate_qwen_local_response(group_path, all_question_tmp, "Qwen2-VL-72B-Instruct", bool_list, i + 1)
                # generate_qwen_local_response(group_path, all_question_tmp, "Qwen2-VL-72B-Instruct-GPTQ-Int4", bool_list, i + 1)
                # get_llava_ov_response(group_path, all_question_tmp, "llava-onevision-qwen2-72b-ov-hf", bool_list, i + 1)
                # get_llava_response(group_path, all_question_tmp, 'llava-v1.6-34b-hf', bool_list, i+1)
                # get_internvl15_response(group_path, all_question_tmp, "InternVL2-Llama3-76B", bool_list, i + 1)
