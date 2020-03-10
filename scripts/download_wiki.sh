# Script to download and preprocess Wikipedia dumps for 17 languages.
# URLs from: https://dumps.wikimedia.org/backup-index.html
# To run: sh download_wiki.sh
set -e

wget https://dumps.wikimedia.org/dewiki/20200301/dewiki-20200301-pages-articles-multistream1.xml-p1p262468.bz2
bzip2 -d dewiki*
python ~/Documents/wikiextractor/WikiExtractor.py dewiki* -o de
rm -rf dewiki*

wget https://dumps.wikimedia.org/zhwiki/20200220/zhwiki-20200220-pages-articles-multistream1.xml-p1p162886.bz2
bzip2 -d zhwiki*
python ~/Documents/wikiextractor/WikiExtractor.py zhwiki* -o zh
rm -rf zhwiki*

wget https://dumps.wikimedia.org/jawiki/20200220/jawiki-20200220-pages-articles-multistream1.xml-p1p106178.bz2
bzip2 -d jawiki*
python ~/Documents/wikiextractor/WikiExtractor.py jawiki* -o ja
rm -rf jawiki*

wget https://dumps.wikimedia.org/eswiki/20200220/eswiki-20200220-pages-articles-multistream1.xml-p1p143637.bz2
bzip2 -d eswiki*
python ~/Documents/wikiextractor/WikiExtractor.py eswiki* -o es
rm -rf eswiki*

wget https://dumps.wikimedia.org/itwiki/20200220/itwiki-20200220-pages-articles-multistream1.xml-p1p277091.bz2
bzip2 -d itwiki*
python ~/Documents/wikiextractor/WikiExtractor.py itwiki* -o it
rm -rf itwiki*

wget https://dumps.wikimedia.org/rowiki/20200220/rowiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d rowiki*
python ~/Documents/wikiextractor/WikiExtractor.py rowiki* -o ro
rm -rf rowiki*

wget https://dumps.wikimedia.org/lawiki/20200220/lawiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d lawiki*
python ~/Documents/wikiextractor/WikiExtractor.py lawiki* -o la
rm -rf lawiki*

wget https://dumps.wikimedia.org/nowiki/20200220/nowiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d nowiki*
python ~/Documents/wikiextractor/WikiExtractor.py nowiki* -o no
rm -rf nowiki*

wget https://dumps.wikimedia.org/dawiki/20200220/dawiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d dawiki*
python ~/Documents/wikiextractor/WikiExtractor.py dawiki* -o da
rm -rf dawiki*

wget https://dumps.wikimedia.org/nlwiki/20200220/nlwiki-20200220-pages-articles-multistream1.xml-p1p123351.bz2
bzip2 -d nlwiki*
python ~/Documents/wikiextractor/WikiExtractor.py nlwiki* -o nl
rm -rf nlwiki*

wget https://dumps.wikimedia.org/slwiki/20200220/slwiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d slwiki*
python ~/Documents/wikiextractor/WikiExtractor.py slwiki* -o sl
rm -rf slwiki*

wget https://dumps.wikimedia.org/plwiki/20200220/plwiki-20200220-pages-articles-multistream1.xml-p1p169750.bz2
bzip2 -d plwiki*
python ~/Documents/wikiextractor/WikiExtractor.py plwiki* -o pl
rm -rf plwiki*

wget https://dumps.wikimedia.org/arwiki/20200220/arwiki-20200220-pages-articles-multistream1.xml-p1p186249.bz2
bzip2 -d arwiki*
python ~/Documents/wikiextractor/WikiExtractor.py arwiki* -o ar
rm -rf arwiki*

wget https://dumps.wikimedia.org/hewiki/20200220/hewiki-20200220-pages-articles-multistream1.xml-p1p54634.bz2
bzip2 -d hewiki*
python ~/Documents/wikiextractor/WikiExtractor.py hewiki* -o he
rm -rf hewiki*

wget https://dumps.wikimedia.org/idwiki/20200220/idwiki-20200220-pages-articles-multistream.xml.bz2
bzip2 -d idwiki*
python ~/Documents/wikiextractor/WikiExtractor.py idwiki* -o id
rm -rf idwiki*
