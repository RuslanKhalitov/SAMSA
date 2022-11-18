wget https://storage.googleapis.com/long-range-arena/lra_release.gz
gzip -d lra_release.gz
tar xvf lra_release
wget https://dl.fbaipublicfiles.com/mega/data/lra.zip
unzip lra.zip
rm lra.zip
bash create_datasets.sh

