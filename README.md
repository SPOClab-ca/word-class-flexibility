# Word Class Flexibility (EMNLP 2020)

Source code and data for paper: "*Word class flexibility: A deep contextualized approach*" under review at EMNLP 2020.

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

