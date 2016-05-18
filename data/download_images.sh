# cat urls.txt | awk '{ print $2 }' | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./train/
head -1 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./train/
head -2 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./val/
head -3 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./test/
head -4 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./paintings/