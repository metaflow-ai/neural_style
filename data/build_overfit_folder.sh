DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SIZE=600
OVERFITDIR=$DIR/overfit_$SIZE 
OUTPUTDIR=$OVERFITDIR/cv

mkdir -p $OVERFITDIR
mkdir -p $OUTPUTDIR

for file in $(ls $DIR/val | grep jpg | head -200)
do
    convert $DIR/val/$file -resize ${SIZE}x${SIZE}\! $OVERFITDIR/$file
done
python gatys_paper.py --content $OVERFITDIR --image_size $SIZE --max_iter 500 --no_dump_losses 1 --output_dir $OUTPUTDIR --alpha 100 --beta 1 --gamma 0.00005
for file in $(ls $OUTPUTDIR/ | grep jpg)
do
    mv $OUTPUTDIR/$file $OUTPUTDIR/${file/_contentconv**\.jpg/.jpg}
done
