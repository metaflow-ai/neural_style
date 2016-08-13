head -1 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5
unzip train2014.zip && mv train2014 train 
head -2 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5
unzip val2014.zip && mv val2014 val
head -3 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5
unzip test2014.zip && mv test2014 test
head -4 urls.txt | tail -1 | xargs wget --no-check-certificate -t 1 --timeout=5 -P paintings/
cd ./paintings && unzip paintings.zip

# Dump preprocess paintings for 600x600 images by default
python prepare_data.py