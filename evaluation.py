import json

def load_ground_truth_map(inputFile):
    questionToPIDs = {}
    with open(inputFile, 'r') as f:
        for line in f:
            item = json.loads(line)
            questionToPIDs[item['question']] = item['pids']
    return questionToPIDs

def create_question_line_map(file_path):
    question_line_map = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines, start=0):
            question = line.split("\t")[0].strip()
            question_line_map[question] = line_number
    return question_line_map

def clean_results(questionToPIDs, question_line_map, predictions):
    new_preds = []
    ground_truth = []
    
    for q, lineNum in question_line_map.items():
        if q in questionToPIDs:
            new_preds.append(predictions[lineNum])
            ground_truth.append(questionToPIDs[q])
    print(len(ground_truth), len(new_preds))
    return ground_truth, new_preds


def load_predictions(filename):
    predictions = []
    with open(filename, 'r') as f:
        for line in f:
            predictions.append(line.strip().split(','))
    return predictions

def precision_at_k(ranked_list, ground_truth, k):
    relevant_count = 0
    for i in range(k):
        if ranked_list[i] in ground_truth:
            relevant_count += 1
    return relevant_count / k

def average_precision(ranked_list, ground_truth):
    relevant_count = 0
    sum_precision = 0
    for i, item in enumerate(ranked_list):
        if item in ground_truth:
            relevant_count += 1
            sum_precision += relevant_count / (i + 1)
    if relevant_count == 0:
        return 0
    return sum_precision / relevant_count

def mean_average_precision(ground_truth, predictions):
    total_ap = 0
    num_questions = len(ground_truth)
    for gt, pred in zip(ground_truth, predictions):
        total_ap += average_precision(pred, gt)
    return total_ap / num_questions

questionToPIDs = load_ground_truth_map('/home/akashp/145/OAG-AQA/data/AQA/qa_train.txt')
# print(questionToPIDs)
question_line_map = create_question_line_map('/home/akashp/145/OAG-AQA/data/kddcup/dpr/qa_valid_dpr.tsv')
# print(question_line_map)
predictions = load_predictions('/home/akashp/145/OAG-AQA/roberta_out_23.txt')

ground_truth, predictions = clean_results(questionToPIDs, question_line_map, predictions)

map_score = mean_average_precision(ground_truth, predictions)
print(f'Mean Average Precision (MAP): {map_score}')

def top_k_average_precision(ranked_list, ground_truth, k):
    relevant_count = 0
    sum_precision = 0
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i] in ground_truth:
            relevant_count += 1
            sum_precision += relevant_count / (i + 1)
    if relevant_count == 0:
        return 0
    return sum_precision / relevant_count

def top_k_mean_average_precision(ground_truth, predictions, k):
    total_ap = 0
    num_questions = len(ground_truth)
    for gt, pred in zip(ground_truth, predictions):
        total_ap += top_k_average_precision(pred, gt, k)
    return total_ap / num_questions

top_k = 20
top_k_map_score = top_k_mean_average_precision(ground_truth, predictions, top_k)
print(f'Top-{top_k} Mean Average Precision (MAP): {top_k_map_score}')