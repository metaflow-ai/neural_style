MAINDIR='st'
python predict.py --models_dir $PWD/models/data/$MAINDIR
for dir in $(ls -d data/output/$MAINDIR/*)
do
    rm -f $PWD/$dir/mosaic.png
    montage $(ls -1 $PWD/$dir/* | grep '.*png' | xargs) -tile 10x9 -geometry +0+0 $PWD/$dir/mosaic.png
    # rm -f $dir/\d\d\d\d.*
done