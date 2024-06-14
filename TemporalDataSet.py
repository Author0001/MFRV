import torch
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import TensorDataset 
from pairwise_ffnn_pytorch import VerbNet
from utils import LabelType

class bigramGetter_fromNN:
    def __init__(self, device, emb_path, mdl_path, ratio=0.3, layer=1, emb_size=200, splitter=','):
        self.verb_i_map = {}
        self.device = device
        f = open(emb_path)
        lines = f.readlines()
        for i, line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        f.close()
        self.model = VerbNet(
            len(self.verb_i_map), hidden_ratio=ratio, emb_size=emb_size, num_layers=layer)
        self.model.to(self.device)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self, v1, v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1], self.verb_i_map[v2]]])).to(self.device))

    def getBigramStatsFromTemprel(self, temprel):
        if type(temprel.lemma) == type((0, 1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1, v2 = '', ''
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.tensor([0, 0]).view(1, -1).to(self.device)
        return torch.cat((self.eval(v1, v2), self.eval(v2, v1)), 1).view(1, -1)

    def retrieveEmbeddings(self, temprel):
        if type(temprel.lemma) == type((0, 1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1, v2 = '', ''
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0, 0]])).to(self.device)).view(1, -1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1], self.verb_i_map[v2]]])).to(self.device)).view(1, -1)






class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.text = ""
        self.event_offset = []

        is_start = True
        for i,d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])

            if is_start:
                is_start = False
            else:
                self.text += " "
            
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            
            self.text += tmp[0]

        assert len(self.event_ix) == 2


class temprel_set:
    def __init__(self, xmlfname, datasetname="matres"):
        self.xmlfname = xmlfname
        self.datasetname = datasetname

        self.bigramStats_dim = 1
        self.granularity = 0.5
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))
    
    def to_tensor(self, tokenizer):
        gathered_text = [ee.text for ee in self.temprel_ee]
        
        tokenized_output = tokenizer(gathered_text, padding=True, return_offsets_mapping=True)
        tokenized_event_ix = []


        ####3
        ratio = 0.3
        emb_size = 200
        layer = 1
        splitter = " "
        print("---------")
        print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
        emb_path = './ser/embeddings_%.1f_%d_%d_timelines.txt' % (
            ratio, emb_size, layer)
        mdl_path = './ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
        bigramGetter = bigramGetter_fromNN(
            'cpu', emb_path, mdl_path, ratio, layer, emb_size, splitter=splitter)


        current_batch = {'commonsense': []}
        for i in range(len(self.temprel_ee)):
            event_ix_pair = []
            for j, offset_pair in enumerate(tokenized_output['offset_mapping'][i]):
                if (offset_pair[0] == self.temprel_ee[i].event_offset[0] or\
                    offset_pair[0] == self.temprel_ee[i].event_offset[1]) and\
                   offset_pair[0] != offset_pair[1]:
                    event_ix_pair.append(j)
            if len(event_ix_pair) != 2:
                raise ValueError(f'Instance {i} doesn\'t found 2 event idx.')
            tokenized_event_ix.append(event_ix_pair)
        for i,temprel in enumerate(self.temprel_ee):
            # common sense embeddings
            bigramstats = bigramGetter.getBigramStatsFromTemprel(
                    temprel).detach().cpu().numpy()
            #print(temprel)
            commonsense = [min(int(1.0 / self.granularity) - 1,
                                   int(bigramstats[0][0] / self.granularity))]
            for k in range(1, self.bigramStats_dim):
                commonsense.append((k - 1) * int(1.0 / self.granularity) +
                                       min(int(1.0 / self.granularity) - 1,
                                           int(bigramstats[0][k] / self.granularity)))
            current_batch['commonsense'].append(commonsense)

        common_ids = torch.LongTensor(
                    current_batch['commonsense'])
                # [1,common_sense_emb_dim+bigramStats_dim]
        input_ids = torch.LongTensor(tokenized_output['input_ids'])
        attention_mask = torch.LongTensor(tokenized_output['attention_mask'])
        tokenized_event_ix = torch.LongTensor(tokenized_event_ix)
        


        labels = torch.LongTensor([LabelType.to_class_index(ee.label) for ee in self.temprel_ee])
        return TensorDataset(input_ids, attention_mask, tokenized_event_ix, labels,common_ids)