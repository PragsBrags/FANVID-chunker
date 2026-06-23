The script is designed to be run from the command line and requires four mandatory inputs:

python split_fanvid.py \
    --dataset-csv dataset_lp.csv \
    --annotations-csv license_plate_annotations_HR.csv \
    --videos-dir videos \
    --output-dir output \
    --clip-length 2
