import os

entities_file_path = "./data/subjects_of_interests.data"
context_file_path = "./data/context_of_disease_which_has_relation_to_genes.csv"
extracted_entity_log_path = "./data/extracted_entity.log"
extracted_context_log_path = "./data/extracted_context.log"

with open(entities_file_path, 'r') as file:
	raw_entities = file.read()
entities = raw_entities.lower().split("\n")
entities = list(dict.fromkeys(entities))

with open(context_file_path, 'r') as file:
	raw_contexts = file.read()
contexts = raw_contexts.lower().split(".")

extracted_contexts = []
context_counts = [0 for _ in entities]

for i in range(len(contexts)):
	for j in range(len(entities)):
		if entities[j] in contexts[i]:
			extracted_contexts.append(contexts[i])
			context_counts[j] += 1
			break

entity_log = ""
for i in range(len(entities)):
	entity_log += f"{entities[i]}, {context_counts[i]}" + "\n"
with open(extracted_entity_log_path, "w") as entity_log_file:
	entity_log_file.write(entity_log)

context_log = ""
for context in extracted_contexts:
	context_log += context + "\n"
with open(extracted_context_log_path, "w") as context_log_file:
	context_log_file.write(context_log)
print(f"total {len(contexts)} relations before extraction")
print(f"total {len(extracted_contexts)} relations after extraction")
print(f"total {len(raw_contexts)} characters before extraction")
print(f"total {len(context_log)} characters after extraction")


