# cat urls.txt | awk '{ print $2 }' | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./train/
head -1 data/urls.txt | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./train/
head -2 data/urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./val/
head -3 data/urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./test/