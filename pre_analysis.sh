python layer_reconstruction.py
montage $(ls -d1 $PWD/data/output/vgg19/reconstruction/* | grep '\(content\|style\)_conv.*png' | xargs) -tile 16x2 -geometry +0+0 data/output/vgg19/reconstruction/layer_reconstruction.png
# rm -f data/output/vgg19/reconstruction/.*(content|style)_conv.*png

python ltv.py
montage $(ls -d1 $PWD/data/output/vgg19/ltv/* | grep '_gamma.*png' | xargs) -tile 6x2 -geometry +0+0 data/output/vgg19/ltv/layer_ltv.png
# rm -f data/output/vgg19/ltv/.*(content|style)_conv.*png

python alpha.py
montage $(ls -d1 $PWD/data/output/vgg19/alpha/* | grep 'styleconv.*png' | xargs) -tile 6x2 -geometry +0+0 data/output/vgg19/alpha/layer_alpha.png
# rm -f data/output/vgg19/alpha/.*(content|style)_conv.*png

python layer_influence.py
montage $(ls -d1 $PWD/data/output/vgg19/influence/* | grep '_style.*_content.*png' | xargs) -tile 12x12 -geometry +0+0 data/output/vgg19/influence/layer_influence.png
# rm -f data/output/vgg19/influence/.*(content|style)_conv.*png