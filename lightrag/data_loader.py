def _load_biomixqa_true_false(data):
  rows = data.split('\n')
  qa = []
  for row in rows:
    if row == "text,label" or len(row) == 0:
      continue
    pos = row.rfind(',')
    qa.append([row[:pos].replace('"', ''), row[pos+1:].lower()])
  return qa

def get_true_false_data(file_path):
  with open(file_path, 'r') as file:
    file_content = file.read()
  return _load_biomixqa_true_false(file_content)
