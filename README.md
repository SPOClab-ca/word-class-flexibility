# Word Class Flexibility

This repository contains the source code and data for our EMNLP 2020 paper: ["*Word class flexibility: A deep contextualized approach*"](https://arxiv.org/abs/2009.09241) by Bai Li, Guillaume Thomas, Yang Xu, and Frank Rudzicz.

**Abstract**: Word class flexibility refers to the phenomenon whereby a single word form is used across different grammatical categories. Extensive work in linguistic typology has sought to characterize word class flexibility across languages, but quantifying this phenomenon accurately and at scale has been fraught with difficulties. We propose a principled methodology to explore regularity in word class flexibility. Our method builds on recent work in contextualized word embeddings to quantify semantic shift between word classes (e.g., noun-to-verb, verb-to-noun), and we apply this method to 37 languages. We find that contextualized embeddings not only capture human judgment of  class variation within words in English, but also uncover shared tendencies in class flexibility across languages. Specifically, we find greater semantic variation when flexible lemmas are used in their dominant word class, supporting the view that word class flexibility is a directional process. Our work highlights the utility of deep contextualized models in linguistic typology.

## Citation

Please cite this work if you find it useful for your research.

Li, B., Thomas, G., Xu, Y., and Rudzicz, F. (2020) Word class flexibility: A deep contextualized approach. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

```
@inproceedings{li2020wordclass,
  title={Word class flexibility: A deep contextualized approach},
  author={Li, Bai and Thomas, Guillaume and Xu, Yang and Rudzicz, Frank},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```

## Reproducing results in the paper

The following instructions reproduce the main results (Table 2) for English.

1. Create a new virtualenv

   ```
   python3.7 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   git clone https://github.com/attardi/wikiextractor
   ```

3. Download Universal Dependencies

   ```
   wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
   mkdir data/ud_all
   tar xf ud-treebanks-v2.5.tgz --directory data/ud_all
   ```

4. Download and extract English Wikipedia

   ```
   mkdir data/wiki
   cd data/wiki
   wget https://dumps.wikimedia.org/enwiki/20200220/enwiki-20200220-pages-articles-multistream1.xml-p1p30303.bz2
   bzip2 -d enwiki*
   python ../../wikiextractor/WikiExtractor.py enwiki* -o en
   cd ../..
   ```

5. Preprocess Wikipedia (about 4 hours)

   ```
   mkdir data/wiki/processed_udpipe
   PYTHONPATH=. python scripts/process_wikipedia.py \
   	--wiki_dir=data/wiki/ \
   	--ud_dir=data/ud_all/ud-treebanks-v2.5/ \
   	--dest_dir=data/wiki/processed_udpipe/ \
   	--lang=en \
   	--model=udpipe \
   	--tokens 10000000
   ```

7. Run semantic metrics

    ```
    mkdir results
    PYTHONPATH=. python scripts/multilingual_bert_contextual.py \
        --pkl_dir data/wiki/processed_udpipe/ \
        --pkl_file en.pkl \
        --results_dir results/
    ```
