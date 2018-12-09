#-*- coding:utf-8 -*-
import os
import sys
import codecs
import pickle
import logging


def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def compute(inp, oup, logger):
    cnt_file = 0
    for filename in os.listdir(inp):
        data_path1 = os.path.join(inp, filename)
        data_path2 = oup +'example_'+ str(cnt_file) + '.pkl'
        data = {}
        entity,abstract,article,label = [],[],[],[]
        cnt = 0
        with codecs.open(data_path1, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                if line == '\n':
                    cnt += 1
                    continue
                if cnt == 0:
                    pass
                if cnt == 1:
                    article.append(line.replace('\t\t\t', '').replace('\n', ''))
                if cnt == 2:
                    abstract.append(line.replace('\n', '').replace('*', ''))
                if cnt == 3:
                    entity.append(line.replace('\n', ''))
        for idx, sent in enumerate(article):
            if sent[-1] == '1':
                label.append(idx)
        article = [sent[:len(sent)-1] for idx, sent in enumerate(article)]
        entity_dict = {}
        if len(entity) != 0:
            for pair in entity:
                key = pair.split(':')[0]
                value = pair.split(':')[1]
                entity_dict[key] = value
        data['entity'] = entity_dict
        data['abstract'] = abstract
        data['article'] = article
        data['label'] = label
        save(data_path2, data)
        cnt_file += 1
        if cnt_file % 500 == 0:
            logger.info("running the script, extract %d examples already..." % cnt_file)
    logger.info("extract %d examples totally this time, done." % (cnt_file+1))

if __name__ == "__main__":

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print("Using: python prepro.py ./source_dir/ ./target_dir/")
        sys.exit(1)
    inp, oup = sys.argv[1:3]

    compute(inp, oup, logger)

