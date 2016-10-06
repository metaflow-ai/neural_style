DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SIZE=600
OVERFITDIR=$DIR/overfit_$SIZE 
OVERFITLABELSDIR=$OVERFITDIR/labels
CVDIR=$OVERFITDIR/cv
CVLABELSDIR=$CVDIR/labels

mkdir -p $OVERFITLABELSDIR
mkdir -p $CVLABELSDIR

# Training files
for file in $(ls $DIR/train | grep jpg | head -16)
do
    convert $DIR/train/$file -resize ${SIZE}x${SIZE}\! $OVERFITDIR/$file
done
python gatys_paper.py --content $OVERFITDIR --image_size $SIZE --max_iter 500 --no_dump_losses 1 --output_dir $OVERFITLABELSDIR --alpha 100 --beta 1 --gamma 0.00005
for file in $(ls $OVERFITLABELSDIR/ | grep jpg)
do
    mv $OVERFITLABELSDIR/$file $OVERFITLABELSDIR/${file/_contentconv**\.jpg/.jpg}
done

# Cross validation files
for file in $(ls $DIR/val | grep jpg | head -16)
do
    convert $DIR/val/$file -resize ${SIZE}x${SIZE}\! $CVDIR/$file
done
python gatys_paper.py --content $CVDIR --image_size $SIZE --max_iter 500 --no_dump_losses 1 --output_dir $CVLABELSDIR --alpha 100 --beta 1 --gamma 0.00005
for file in $(ls $CVLABELSDIR/ | grep jpg)
do
    mv $CVLABELSDIR/$file $CVLABELSDIR/${file/_contentconv**\.jpg/.jpg}
done
