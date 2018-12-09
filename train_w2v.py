import os
import sys
import multiprocessing
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 4:
        print("Using: python train_w2v.py one-billion-word-benchmark output_gensim_model output_word_vector")
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    model = Word2Vec(LineSentence(inp), size=150, window=6, min_count=2, workers=(multiprocessing.cpu_count()-2), hs=1, sg=1, negative=10)

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=True)


