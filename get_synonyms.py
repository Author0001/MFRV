from nltk.corpus import wordnet
import random
import nltk
nltk.download('wordnet')
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

def synonym_replacement(word):
    synonyms = get_synonyms(word)
    if synonyms:
        return random.choice(list(synonyms))
    else:
        return word

def replace_synonyms_in_line(line):
    parts = line.split(',')
    replaced_e1 = synonym_replacement(parts[0].split('=')[1])
    replaced_e2 = synonym_replacement(parts[1].split('=')[1])
    return f"E1={replaced_e1},E2={replaced_e2},{','.join(parts[2:])}"

# 读取数据集
with open('/home/JJJ/MFRV/data/train.txt', 'r') as file:
    dataset_lines = file.readlines()

# 对每一行进行同义词替换
modified_lines = [replace_synonyms_in_line(line) for line in dataset_lines]

# 将替换后的内容写入新文件
with open('/home/JJJ/MFRV/data/modified_train.txt', 'w') as file:
    file.writelines(modified_lines)
