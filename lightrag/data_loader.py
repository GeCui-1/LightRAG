def _load_biomixqa_true_false(file_content):
    rows = file_content.split('\n')
    qa = []
    for row in rows:
        if row == "text,label" or len(row) == 0:
            continue
        pos = row.rfind(',')
        qa.append([row[:pos].replace('"', ''), row[pos+1:].lower()])
    return qa

def get_true_false_data(dataset_path):
    with open(dataset_path, 'r') as file:
        file_content = file.read()
    return _load_biomixqa_true_false(file_content)

def _load_biomixqa_mcq(file_content):
    questions = []
    answers = []
    rows = file_content.split('\n')
    question = ""
    answer = ""
    for i in range(len(rows)):
        row = rows[i].lower()
        if len(row) == 0:
            continue
        if i % 2 == 0:
            question = row
            questions.append(question)
        else:
            answer = row.split(':')[1].replace(' ', '')
            answers.append(answer)
    assert len(questions) == len(answers)
    return (questions, answers)

def get_mcq_data(dataset_path):
    with open(dataset_path, 'r') as file:
        file_content = file.read()
    return _load_biomixqa_mcq(file_content)
