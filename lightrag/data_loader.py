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
