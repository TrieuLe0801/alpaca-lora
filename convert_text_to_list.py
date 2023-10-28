def convert_text_to_list(text_path: str = "public_test_questions_en.txt"):
    if text_path.endswith(".txt"):
        with open(text_path, 'r') as file:
            content = file.read()

        # Split by Question:
        sections = content.split('Question:')

        # Remove white-space
        sections = [section.strip() for section in sections if section.strip()]
        
        formatted_sections = []

        for section in sections:
            question, answer = section.split('Answer:')
            formatted_question = f"Question: {question.strip()}"
            formatted_answer = f"Answer:{answer.strip()}"
            formatted_sections.append(formatted_question+"\n"+formatted_answer)
            print(formatted_question+"\n"+formatted_answer)
            print("="*18)

        # for section in formatted_sections:
        #     print(section)
    else:
        print("Only convert text file")
    return formatted_sections
