import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.data_loader import get_true_false_data
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

# meta data for the test
num_questions = 311  # totally 311 questions
query_mode = 'naive' # choose from naive, local, global, hybrid

WORKING_DIR = "./biomixqa_tf"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

context_file_path = "./data/extracted_context.log"
qa_file_path = "./data/true_false_biomix.data"


# create RAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)
with open(context_file_path, "r", encoding="utf-8") as f:
    context = f.read()
    print(context)
    rag.insert(context)

# Load questions and answers
true_false_biomix_qa = get_true_false_data(qa_file_path)
# print(true_false_biomix_qa)

correct = 0
wrong = 0
unsure = 0
for i in range(num_questions):
    print("")
    print("instance number ", i)
    
    question = true_false_biomix_qa[i][0] + ". Answer should start with true, false, or unsure."
    label = true_false_biomix_qa[i][1]
    ans = rag.query(question, param=QueryParam(mode=query_mode))
  
    print("question: ", question)
    print("label: ", label)
    print("ans: ", ans)
  
    if "unsure" in ans.lower():
        print("LLM is unsure about this question")
        unsure += 1
    elif label in ans.lower():
        print("answer is correct")
        correct += 1
    else:
        print("answer is wrong")
        wrong += 1

print("unsure count: ", unsure)
print("correct count: ", correct)
print("wrong count: ", wrong)
