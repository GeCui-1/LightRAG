import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.data_loader import get_true_false_data
import time
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

# meta data for the test
num_questions = 1000  # totally 311 questions
query_mode = 'naive' # choose from naive, local, global, hybrid

WORKING_DIR = "./tf"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

context_file_path = "./data/treatment/extracted_mcq_context.log"

#################################################################### create RAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)
with open(context_file_path, "r", encoding="utf-8") as f:
    raw_contexts = f.read()
contexts = raw_contexts.split("\n")
buffer = ""
print("context length:", len(contexts))
for i in range(len(contexts)):
    buffer += contexts[i] + "\n"
    if i % 100 == 99:
        rag.insert(buffer)
        time.sleep(1)
        buffer = ""
        break  # for debugging only
if len(buffer) > 0:
    # print("buffer:", buffer)
    rag.insert(buffer)
#################################################################### create RAG

# Load questions and answers
dataset_paths = {
    "cancer": "./data/mcq/mcq_cancer.data",
    "gene": "./data/mcq/mcq_genetic_disorder.data",
    "heart": "./data/mcq/mcq_heart.data",
    "immune": "./data/mcq/mcq_immune.data",
    "infection": "./data/mcq/mcq_infection.data",
    "neural": "./data/mcq/mcq_neural.data",
}

qas = {}
corrects = {}
wrongs = {}
unsures = {}
exceptions = {}
total_times = {}
for key, val in dataset_paths.items():
    questions, answers = get_mcq_data(dataset_paths[key])
    qas[key] = (questions, answers)
    corrects[key] = 0
    wrongs[key] = 0
    unsures[key] = 0
    exceptions[key] = 0
    total_times[key] = 0.0

for qa_type, qa in qas.items():
    print("qa type:", qa_type)
    questions = qa[0]
    labels = qa[1]
    print("questions:", questions)
    print("answers:", answers)
    for i in range(min(num_questions, len(questions))):
        print("")
        print("instance number ", i)
        
        question = questions[i] + ". Answer should start with true, false, or unsure."
        label = labels[i].lower()
        start_time = time.time()
        
        ans = rag.query(question, param=QueryParam(mode=query_mode))
        
        end_time = time.time()
        total_times[qa_type] += end_time - start_time
    
        print("question: ", question)
        print("label: ", label)
        print("ans: ", ans)
      
        if label in ans.lower():
            print("answer is correct")
            corrects[qa_type] += 1
        else:
            print("answer is wrong")
            wrongs[qa_type] += 1

print("correct count: ", corrects)
print("wrong count: ", wrongs)
print("total time: ", total_times)
