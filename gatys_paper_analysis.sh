python gatys_paper.py --pooling_type max --input_type random
for folder in $(ls -d data/output/vgg19/gatys_max_random*)
do
  if [ "$(ls -A $PWD/$folder)" ]; then
    montage $(ls -d1 $PWD/$folder/* | grep '_contentconv.*png' | xargs) -tile 3x3 -geometry +0+0 $PWD/$folder/mosaic.png
    # rm -f $PWD/$folder/.*_contentconv.*png
  fi
done

python gatys_paper.py --pooling_type avg --input_type random
for folder in $(ls -d data/output/vgg19/gatys_avg_random*)
do
  if [ "$(ls -A $PWD/$folder)" ]; then
    montage $(ls -d1 $PWD/$folder/* | grep '_contentconv.*png' | xargs) -tile 3x3 -geometry +0+0 $PWD/$folder/mosaic.png
    # rm -f $PWD/$folder/.*_contentconv.*png
  fi
done

python gatys_paper.py --pooling_type max --input_type content
for folder in $(ls -d data/output/vgg19/gatys_max_content*)
do
  if [ "$(ls -A $PWD/$folder)" ]; then
    montage $(ls -d1 $PWD/$folder/* | grep '_contentconv.*png' | xargs) -tile 3x3 -geometry +0+0 $PWD/$folder/mosaic.png
    # rm -f $PWD/$folder/.*_contentconv.*png
  fi
done

python gatys_paper.py --pooling_type avg --input_type content
for folder in $(ls -d data/output/vgg19/gatys_avg_content*)
do
  if [ "$(ls -A $PWD/$folder)" ]; then
    montage $(ls -d1 $PWD/$folder/* | grep '_contentconv.*png' | xargs) -tile 3x3 -geometry +0+0 $PWD/$folder/mosaic.png
    # rm -f $PWD/$folder/.*_contentconv.*png
  fi
done

# for file in $(ls data/overfit | grep jpg)
# do
#     python gatys_paper.py --content data/overfit/$file --pooling_type avg --input_type content --image_size 600 --print_inter_img 1
# done