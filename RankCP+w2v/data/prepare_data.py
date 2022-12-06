import os, json, pickle

data_dir = "../../data_combine_eng/"
output_dir = "split10"

def load_data(input_file, e_cls):
    print('load data_file: {}'.format(input_file))
    data = []
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = line[0]
        doc_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        pos, cause = zip(*pairs)
        pairs = [[p, c] for p, c in zip(pos, cause)]
        clauses = []
        for i in range(doc_len):
            line = inputFile.readline().strip().split(',')
            clause_id, e_cat, e_token = line[0], line[1], line[2]
            clause = ','.join(line[3:])
            clauses.append({
                'clause_id': clause_id, 
                'emotion_category': e_cat,
                'emotion_token': e_token,
                'clause': clause
            })
        data.append({
            'doc_id': doc_id, 'doc_len': doc_len,
            'pairs': pairs, 'clauses': clauses
        })
        e_cls[doc_id] = [p for p in pos]
    return data, e_cls

def write_json(input_file, data):
    output_file = input_file.split('.')[0] + ".json"
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(json.dumps(data))
    print("Writing json file: {}".format(os.path.join(output_dir, output_file)))

def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


if __name__ == '__main__':
    # print(os.listdir(data_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    e_cls = {}
    for filename in os.listdir(data_dir):
        if filename.find("fold") != -1:
            data, e_cls = load_data(data_dir+filename, e_cls)
            write_json(filename, data)
    write_b(e_cls, 'sentimental_clauses_eng.pkl')