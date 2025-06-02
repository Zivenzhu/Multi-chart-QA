import json
import os
from natsort import natsorted
import chardet
from mmmu_utils import eval_multi_choice, eval_open, parse_open_response, parse_multi_choice_response


def open_json_file(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data, encoding

## The following function is used for error analysis, which is not used in the main evaluation
# def convert_to_markdown(error_folders):
#     markdown_lines = []
#     for group_name, error_details in error_folders.items():
#         markdown_lines.append(f"## Group: {group_name}")
#         for error in error_details:
#             for question_num, detail in error.items():
#                 markdown_lines.append(f"### Question {question_num}")
#                 markdown_lines.append(f"- **Ground Truth:** {detail['Ground_truth']}")
#                 markdown_lines.append(f"- **Solution:** {detail['Solution']}")
#                 markdown_lines.append("")  
#         markdown_lines.append("")  
#     return "\n".join(markdown_lines)


bool_list = [0, 0]
tmp_folder_list = [["with_chart_reference", "without_chart_reference", "only_single_image_input", "merged_image"], ["with_CoT", "without_CoT", "text_only"]]

base_folder = os.path.join("evaluation", tmp_folder_list[0][bool_list[0]], tmp_folder_list[1][bool_list[1]])

# # Models used in the main chart
# models = ["claude-3-5-sonnet-20240620", "gpt-4o", "gemini-1.5-pro", "gpt-4o-mini",
#              "MiniCPM-V-2_6", "InternVL2-26B",
#           "Qwen2-VL-7B-Instruct", "InternVL-Chat-V1-5", "llava-onevision-qwen2-7b-ov-hf",
#           "MiniCPM-Llama3-V-2_5", "deepseek-vl-7b-chat", "Eagle-X5-13B-Chat",
#           "llava-v1.6-mistral-7b-hf", "Idefics3-8B-Llama3", "idefics2-8b",
#           "chartgemma", "TinyChart-3B-768", "matcha-chartqa"]

# # Models used in the ablation study
# models = ["claude-3-5-sonnet-20240620", "gpt-4o", "gemini-1.5-pro", "gpt-4o-mini",
#           "MiniCPM-V-2_6", "InternVL2-26B",
#           "Qwen2-VL-7B-Instruct", "InternVL-Chat-V1-5", "llava-onevision-qwen2-7b-ov-hf",
#             "MiniCPM-Llama3-V-2_5", "deepseek-vl-7b-chat", "Eagle-X5-13B-Chat",
#             "llava-v1.6-mistral-7b-hf", "Idefics3-8B-Llama3", "idefics2-8b",
#           "chartgemma", "TinyChart-3B-768", "matcha-chartqa"]

# Please select the models you want to evaluate
models = ["gpt-4o"]

question_content_types = [
        "Direct_Questions/Structure",
        "Direct_Questions/Content",
        "Parallel_Questions/Structure",
        "Parallel_Questions/Mixed",
        "Parallel_Questions/Content",
        "Comparative_Reasoning/Structure",
        "Comparative_Reasoning/Content",
        "Sequential_Reasoning/Content",
    ]


# If you want to evaluate without chart reference, set the partial variable to True.
partial = False
if partial:
    if bool_list[0] == 0:
        with_chart_reference = True
    else:
        with_chart_reference = False
    print(f"with chart reference: {with_chart_reference}")

all_model_average_acc = []

# error_folders = {}

dataset_name = "../data"
for model in models:
    total_num = [0 for _ in range(len(question_content_types))]
    correct_num = [0 for _ in range(len(question_content_types))]
    model_folder_name = f"evaluation_{model}"
    model_folder_path = os.path.join(base_folder, model_folder_name)
    group_names = natsorted(
        [f for f in os.listdir(model_folder_path)],
        key=lambda x: x.lower()
    )

    for group_name in group_names:
        solution_folder_path = os.path.join(model_folder_path, group_name)
        answer_folder_path = os.path.join(dataset_name, group_name)
        answers, answers_encoding = open_json_file(os.path.join(answer_folder_path, 'chart-path_and_question-answer_pair.json'))

        if partial:
            try:
                if bool_list[0] != 0:
                    tmp_path = solution_folder_path
                else:
                    tmp_path = os.path.join(
                                    "evaluation", "with_chart_reference", "with_CoT", model_folder_name,
                                    group_name)
                if os.path.exists(tmp_path):
                    solution_file_names = os.listdir(tmp_path)
                    for solution_file_name in solution_file_names:
                        i = int(solution_file_name.rsplit('.', 1)[0].rsplit('_', 1)[-1]) - 1
                        answer = answers[i]["answer"]

                        if with_chart_reference:
                            solution_file, solution_encoding = open_json_file(os.path.join(
                                "evaluation", "with_chart_reference", "with_CoT", model_folder_name,
                                group_name, solution_file_name))
                        else:
                            solution_file, solution_encoding = open_json_file(os.path.join(solution_folder_path, solution_file_name))
                        if isinstance(solution_file, dict):
                            solution_content = solution_file["choices"][0]["message"]["content"]
                        elif isinstance(solution_file, str):
                            solution_content = solution_file
                        else:
                            raise NotImplementedError

                        tmp_question_content_type = answers[i]["task"]
                        tmp_question_content_index = question_content_types.index(tmp_question_content_type)
                        tmp_question_task_type = answers[i]["type"]
                        if tmp_question_task_type == "multiple-choice":
                            all_choices = ['A', 'B', 'C', 'D']
                            index2ans = {
                                "A": "Answer: A",
                                "B": "Answer: B",
                                "C": "Answer: C",
                                "D": "Answer: D",
                            }
                            chosen = parse_multi_choice_response(solution_content, all_choices, index2ans)
                            eval_answers = answer.split(";")
                            eval_answers = [tmp.strip() for tmp in eval_answers]
                            correct_value = eval_multi_choice(eval_answers, chosen)
                        elif tmp_question_task_type == "open-ended":
                            chosen = parse_open_response(solution_content)
                            correct_value = eval_open(answer, chosen)
                            if correct_value != 1:
                                eval_answers = answer.split(";")
                                eval_answers = [tmp.strip() for tmp in eval_answers]
                                correct_value = eval_open(eval_answers, chosen)
                        else:
                            raise ValueError("Wrong task type")
                        correct_num[tmp_question_content_index] += correct_value
                        total_num[tmp_question_content_index] += 1
                else:
                    continue
            except:
                print(group_name)
                print('opening error')
        else:
            try:
                error_files = []
                for i in range(4):
                    answer = answers[i]["answer"]
                    solution_file_name = f"output_of_question_{i+1}.json"
                    solution_file, solution_encoding = open_json_file(os.path.join(solution_folder_path, solution_file_name))
                    if isinstance(solution_file, dict):
                        solution_content = solution_file["choices"][0]["message"]["content"]
                    elif isinstance(solution_file, str):
                        solution_content = solution_file
                    else:
                        raise NotImplementedError

                    tmp_question_content_type = answers[i]["task"]
                    tmp_question_content_index = question_content_types.index(tmp_question_content_type)
                    tmp_question_task_type = answers[i]["type"]
                    if tmp_question_task_type == "multiple-choice":
                        all_choices = ['A', 'B', 'C', 'D']
                        index2ans = {
                            "A": "Answer: A",
                            "B": "Answer: B",
                            "C": "Answer: C",
                            "D": "Answer: D",
                        }
                        chosen = parse_multi_choice_response(solution_content, all_choices, index2ans)
                        eval_answers = answer.split(";")
                        eval_answers = [tmp.strip() for tmp in eval_answers]
                        correct_value = eval_multi_choice(eval_answers, chosen)
                    elif tmp_question_task_type == "open-ended":
                        chosen = parse_open_response(solution_content)
                        correct_value = eval_open(answer, chosen)

                        if correct_value != 1:
                            eval_answers = answer.split(";")
                            eval_answers = [tmp.strip() for tmp in eval_answers]
                            correct_value = eval_open(eval_answers, chosen)

                    else:
                        raise ValueError("Wrong task type")

                    ## The following code is used for error analysis, which is not used in the main evaluation.
                    # if (correct_value != 1) and ("Direct_Questions" in answers[i]["task"]):
                    #     if group_name not in error_folders:
                    #         error_folders[group_name] = []
                    #     error_detail = {
                    #         f"{i+1}":
                    #             {
                    #                 "Ground_truth": f"{answer}",
                    #                 "Solution": f"{solution_content}"
                    #             }
                    #     }
                    #     error_folders[group_name].append(error_detail)

                    correct_num[tmp_question_content_index] += correct_value
                    total_num[tmp_question_content_index] += 1
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Folder name: {solution_folder_path}")

    accs_raw = [correct_num[i]/total_num[i] if total_num[i] != 0 else None for i in range(len(question_content_types))]
    average_acc = sum(correct_num)/sum(total_num)
    
    average_acc_percentage = round(average_acc * 100, 2)
    all_model_average_acc.append(average_acc_percentage)
    accs = [round(acc * 100, 2) for acc in accs_raw if acc != None]
    print(f"& {average_acc_percentage:.2f} &  & {accs[0]:.2f} & {accs[1]:.2f} &   & {accs[2]:.2f} & {accs[3]:.2f} & {accs[4]:.2f} &   & {accs[5]:.2f} & {accs[6]:.2f} &   & {accs[7]:.2f}  \\\\")
    print(f"& {average_acc_percentage:.2f} | {accs[0]:.2f} | {accs[1]:.2f} | {accs[2]:.2f} | {accs[3]:.2f} | {accs[4]:.2f} | {accs[5]:.2f} | {accs[6]:.2f} | {accs[7]:.2f} | \\\\")
    print(model)
    print(f"Acc for all types: {[f'{acc:.2f}' for acc in accs]}")
    print(f"Average acc: {average_acc_percentage:.2f}")
    
    print("\n")

    ## The following code is used for ablation study: the setting of with_specified_chart_only. 
    # average_acc = sum(correct_num[:2]) / sum(total_num[:2])
    # print(f"& {average_acc_percentage:.2f} &  {accs[0]:.2f} & {accs[1]:.2f} & {accs[2]:.2f} \\\\")
    # print(f"& {round(sum(correct_num[:2]) / sum(total_num[:2]) * 100, 2): .2f} &  {accs[0]:.2f} & {accs[1]:.2f} \\\\")
    # print(f"Acc for the first question: {round(sum(correct_num[:2]) / sum(total_num[:2]) * 100, 2):.2f}")

    ## The following code is used for error analysis, which is not used in the main evaluation.
    # with open(f"{model}_error_folders.json", "w") as f:
    #     json.dump(error_folders, f, indent=4)
    # with open(f"{model}_error_folders.md", "w") as f:
    #     markdown_content = convert_to_markdown(error_folders)
    #     f.write(markdown_content)


# print the average accuracy for each model
print(f"average_acc_by_model: {all_model_average_acc}")

## The following code is used for displaying dataset statistics, which is not used in the main evaluation.
# print('Dataset Statistics: ')
# print(f"The number of questions for each category in question_content_types: {total_num}")
# print(f"The number of all the questions: {sum(total_num)}")
# ratio_for_all = [round(total_num[i]/sum(total_num)*100, 1) for i in range(len(total_num))]
#
# def get_sum(total_num, index):
#     if index >=0 and index <=1:
#         return sum(total_num[:2])
#     elif index >=2 and index <=4:
#         return sum(total_num[2:5])
#     elif index >= 5 and index <= 7:
#         return sum(total_num[5:8])
#     else:
#         return sum(total_num[8:])
#
# ratio_for_sub = [round(total_num[i]/get_sum(total_num,i)*100, 1) if get_sum(total_num,i) != 0 else None for i in range(len(total_num))]
# print(f"The proportion of each question category relative to the total number of questions:\n {ratio_for_all}")
# print(f"The proportion of each question category relative to the total number of questions in its parent category.:\n {ratio_for_sub}")
