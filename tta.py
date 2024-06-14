import random
import nltk
from nltk.corpus import wordnet
#nltk.download('wordnet')
import xml.etree.ElementTree as ET

input_file = "/home/JJJ/MFRV/data/augmentation/train.xml"
#input_file = "/home/JJJ/MFRV/data/testset-temprel.xml"

# 读取XML文件
tree = ET.parse(input_file)
root = tree.getroot()
STOP_WORDS = {'is', 'a', 'for', 'and', 'the', 'in', 'to', 'it'}

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    augmented_words = []

    for word in words:
      
        parts = word.split('///')
       
        text, lemma, pos, event_part = parts

       
        if text.lower() not in STOP_WORDS:
            synonyms = get_synonyms(text)
            if synonyms:
                new_text = random.choice(synonyms)
                new_lemma = new_text  
                augmented_word = '///'.join([new_text, new_lemma, pos, event_part])
                augmented_words.append(augmented_word)
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)

def random_insertion(sentence, p=1):
    words = sentence.split()
    augmented_words = []

    for word in words:
       
        parts = word.split('///')
        text, lemma, pos, event_part = parts

        
        if event_part not in ["E1", "E2"]:
            
            if text.lower() not in STOP_WORDS:
                synonyms = get_synonyms(text)
                if synonyms and random.random() < p:  
                    new_text = random.choice(synonyms)
                    new_lemma = new_text 
                    new_word = '///'.join([new_text, new_lemma, pos, event_part])
                    augmented_words.append(new_word)
                augmented_words.append(word)
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)




def random_swap(sentence, p=0.5):
    words = sentence.split()
    augmented_words = words.copy()

    for i in range(len(words)):
        if random.random() < p:
           
            j = random.choice([idx for idx in range(len(words)) if idx != i])

          
            augmented_words[i], augmented_words[j] = augmented_words[j], augmented_words[i]

    return ' '.join(augmented_words)


def random_deletion(sentence, p=0.8):
    words = sentence.split()
    augmented_words = []

    for word in words:
  
        parts = word.split('///')
        text, lemma, pos, event_part = parts

    
        if event_part in ["E1", "E2"]:
            augmented_words.append(word)
        else:
      
            if random.uniform(0, 1) > p:
           
                augmented_words.append('')
            else:
                augmented_words.append(word)

   
    augmented_words = [word for word in augmented_words if word]

    return ' '.join(augmented_words)



new_root = ET.Element("DATA")


for sentence_elem in root.findall(".//SENTENCE"):

    sentence = sentence_elem.text
    #print(sentence)
    attributes = ' '.join([f'{key}="{value}"' for key, value in sentence_elem.attrib.items()])


    synonym_replaced_sentence = synonym_replacement(sentence)
    inserted_sentence = random_insertion(sentence)
    swapped_sentence = random_swap(sentence)
    deleted_sentence = random_deletion(sentence)


    #new_sentence_elem_1 = ET.SubElement(new_root, "SENTENCE", attrib=sentence_elem.attrib)
    #new_sentence_elem_2 = ET.SubElement(new_root, "SENTENCE", attrib=sentence_elem.attrib)
    #new_sentence_elem_3 = ET.SubElement(new_root, "SENTENCE", attrib=sentence_elem.attrib)
    new_sentence_elem_4 = ET.SubElement(new_root, "SENTENCE", attrib=sentence_elem.attrib)

    #new_sentence_elem_1.text = synonym_replaced_sentence
    #new_sentence_elem_2.text = inserted_sentence
    #new_sentence_elem_3.text = swapped_sentence
    new_sentence_elem_4.text = deleted_sentence
    

# 创建新的XML树
new_tree = ET.ElementTree(new_root)


#output_file_1 = "/home/JJJ/MFRV/data/augmentation/synonym_replacement_middle.xml"
#output_file_2 = "/home/JJJ/MFRV/data/augmentation/inserted_sentence_middle.xml"
#output_file_3 = "/home/JJJ/MFRV/data/augmentation/swapped_sentence_middle.xml"
output_file_4 = "/home/JJJ/MFRV/data/augmentation/deleted_sentence_middle.xml"

#new_tree.write(output_file_1, encoding="utf-8", xml_declaration=True)
#new_tree.write(output_file_2, encoding="utf-8", xml_declaration=True)
#new_tree.write(output_file_3, encoding="utf-8", xml_declaration=True)
new_tree.write(output_file_4, encoding="utf-8", xml_declaration=True)
""" 

with open("/home/JJJ/MFRV/data/augmentation/synonym_replacement_middle.xml", "r") as file:
    data = file.read()

data = data.replace("<SENTENCE>", "\n<SENTENCE>").replace("</SENTENCE>", "</SENTENCE>\n")


with open("/home/JJJ/MFRV/data/augmentation/synonym_replacement.xml", "w") as file:
    file.write(data)
 
with open("/home/JJJ/MFRV/data/augmentation/inserted_sentence_middle.xml", "r") as file:
    data = file.read()

data = data.replace("<SENTENCE>", "\n<SENTENCE>").replace("</SENTENCE>", "</SENTENCE>\n")

with open("/home/JJJ/MFRV/data/augmentation/inserted_sentence.xml", "w") as file:
    file.write(data)
    

with open("/home/JJJ/MFRV/data/augmentation/swapped_sentence_middle.xml", "r") as file:
    data = file.read()

data = data.replace("<SENTENCE>", "\n<SENTENCE>").replace("</SENTENCE>", "</SENTENCE>\n")

with open("/home/JJJ/MFRV/data/augmentation/swapped_sentence.xml", "w") as file:
    file.write(data)
""" 
with open("/home/JJJ/MFRV/data/augmentation/deleted_sentence_middle.xml", "r") as file:
    data = file.read()

data = data.replace("<SENTENCE>", "\n<SENTENCE>").replace("</SENTENCE>", "</SENTENCE>\n")

with open("/home/JJJ/MFRV/data/augmentation/deleted_sentence.xml", "w") as file:
    file.write(data)
 


original_sentence = "The///the///DT///B communication///communication///NN///B ,///,///,///B which///which///WDT///B occurred///occur///VBD///E1 as///as///IN///M President///president///NNP///M Barack///barack///NNP///M Obama///obama///NNP///M concluded///conclude///VBD///E2 his///his///PRP$///A initial///initial///JJ///A presidential///presidential///JJ///A trip///trip///NN///A to///to///TO///A Israel///israel///NNP///A ,///,///,///A was///be///VBD///A an///an///DT///A unforeseen///unforeseen///JJ///A result///result///NN///A of///of///IN///A a///a///DT///A Middle///middle///NNP///A East///east///NNP///A journey///journey///NN///A that///that///WDT///A appeared///appear///VBD///A to///to///TO///A produce///produce///VB///A few///few///JJ///A tangible///tangible///JJ///A actions///action///NNS///A .///.///.///A"
synonym_replaced_sentence = synonym_replacement(original_sentence)
inserted_sentence = random_insertion(original_sentence, p=0.5)
swapped_sentence = random_swap(original_sentence, p=0.5)
deleted_sentence = random_deletion(original_sentence, p=0.8)

print("Original:", original_sentence)
print("Synonym Replacement:", synonym_replaced_sentence)
print("Random Insertion:", inserted_sentence)
print("Random Swap:", swapped_sentence)
print("Random Deletion:", deleted_sentence)
