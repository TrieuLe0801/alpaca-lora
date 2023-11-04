import json

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

answer_mapping = {
    0: ("opa", "1000"),
    1: ("opb", "0100"),
    2: ("opc", "0010"),
    3: ("opd", "0001"),
}


def transform_dataset(
    data_path_or_name: str = "",
    dataset_key: str = "train",
    instruction: str = "",
    output_path: str = "",
):
    """Transform data to the structure:
    {
        "instruction":"",
        "input":"",
        "output":"",
    }

    Args:
        data_path_or_name (str, optional): Data path or name from Huggingface Dataset. Defaults to "".
        dataset_key (str, optional): Key values to get dataset from huggingface
        instruction (str, optional): Instruction for question and answer
        output_path (str, optional): Path to save processed data, including name of file
    """
    if data_path_or_name.endswith(".json") or data_path_or_name.endswith(
        ".jsonl"
    ):
        data = load_dataset("json", data_files=data_path_or_name)
    elif data_path_or_name.endswith(".txt"):
        data = load_dataset("text", data_files=data_path_or_name)
    else:
        # Load data from huggingface
        data = load_dataset(data_path_or_name)
        data = data[dataset_key]

    # print(data["train"])

    # Change all into csv data frame
    data_df = pd.DataFrame(data)

    # Create instruction
    # The instruction is the exlanation
    instructions_list = [instruction] * data_df.shape[0]

    # Create input and output
    # Input includes Question and Options, answer is based on cop column
    # Output has 2 kinds (Textual answer and Binary answer)
    inputs_list = []
    answers_list = []
    b_answers_list = []

    for _, row in tqdm(data_df.iterrows()):
        question = f'Q: {row["question"]}'
        options = "".join(
            [
                f"""'A': '{row["opa"]}'""",
                f""",'B': '{row["opb"]}'""",
                f""",'C': '{row["opc"]}'""",
                f""",'D': '{row["opd"]}'""",
            ]
        )
        inputs_list.append(question+"{"+options+"}")

        cop = row["cop"]
        answer_col, b_answer = answer_mapping.get(cop, ("opd", "0001"))
        answer = row[answer_col]
        answers_list.append(answer)
        # print(b_answer)
        b_answers_list.append(b_answer)

    # Create new columns
    data_df["instruction"] = instructions_list
    data_df["input"] = inputs_list
    data_df["output"] = answers_list
    data_df["binary_output"] = b_answers_list

    # Save file
    if output_path.endswith(".json"):
        data_df.to_json(output_path, orient="records")
    elif output_path.endswith(".csv"):
        data_df.to_csv(output_path, index=False)
    else:
        raise "Invalid path of file!"

# def prepare_dataset():
#     train_dataset = load_dataset("json", data_files="data/medmcqa_data_train.json")["train"]
#     test_dataset = load_dataset("json", data_files="data/medmcqa_data_test.json")["train"]
#     valid_dataset = load_dataset("json", data_files="data/medmcqa_data_valid.json")["train"]
#     print(test_dataset)
#     data_dict = {}
#     data_dict.update({"train": train_dataset, "test": test_dataset, "validation": valid_dataset})
    
#     with open("data/medmcqa_data.json", "w") as outfile: 
#         json.dump(dict(data_dict), outfile)
    
if __name__ == "__main__":
    instruction = """
       "Please answer the letters of options truthfully in the bracket, there maybe more than one correct answers
    """
    transform_dataset(
        data_path_or_name="medmcqa",
        dataset_key="train",
        instruction=instruction,
        output_path="data/medmcqa_data_train.json",
    )
    # prepare_dataset()