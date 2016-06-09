python gatys_paper.py --pooling_type max --input_type random --max_iter 1
montage $(ls -d1 $PWD/data/output/vgg19/gatys_max_random/* | grep '_content.*png' | xargs) -tile 3x3 -geometry +0+0 data/output/vgg19/gatys_max_random/layer_gatys_max_random.png
# rm -f data/output/vgg19/gatys_max_random/.*_content.*png

python gatys_paper.py --pooling_type avg --input_type random --max_iter 1
montage $(ls -d1 $PWD/data/output/vgg19/gatys_avg_random/* | grep '_content.*png' | xargs) -tile 3x3 -geometry +0+0 data/output/vgg19/gatys_avg_random/layer_gatys_avg_random.png
# rm -f data/output/vgg19/gatys_avg_random/.*_content.*png

python gatys_paper.py --pooling_type max --input_type content --max_iter 1
montage $(ls -d1 $PWD/data/output/vgg19/gatys_max_content/* | grep '_content.*png' | xargs) -tile 3x3 -geometry +0+0 data/output/vgg19/gatys_max_content/layer_gatys_max_content.png
# rm -f data/output/vgg19/gatys_max_content/.*_content.*png

python gatys_paper.py --pooling_type avg --input_type content --max_iter 1
montage $(ls -d1 $PWD/data/output/vgg19/gatys_avg_content/* | grep '_content.*png' | xargs) -tile 3x3 -geometry +0+0 data/output/vgg19/gatys_avg_content/layer_gatys_avg_content.png
# rm -f data/output/vgg19/gatys_avg_content/.*_content.*png