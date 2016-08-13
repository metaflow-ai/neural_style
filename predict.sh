if [ $1  ]; then
  MDIR=$1
else
  MDIR='st'
fi

python predict.py --models_dir $PWD/models/data/$MDIR
for dir in $(ls -d data/output/$MDIR/*)
do
    rm -f $PWD/$dir/mosaic.png
    montage $(ls -1 $PWD/$dir/* | grep '.*png' | xargs) -tile 10x9 -geometry +0+0 $PWD/$dir/mosaic.png
    # rm -f $dir/\d\d\d\d.*
done