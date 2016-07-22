# cat urls.txt | awk '{ print $2 }' | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./train/
head -1 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P
unzip train2014.zip && mv train2014 train 
head -2 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P 
unzip val2014.zip && mv val2014 val
head -3 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P 
unzip test2014.zip && mv test2014 test
head -4 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P ./paintings/
cd ./paintings && unzip paintings.zip