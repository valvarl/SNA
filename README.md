Extract yelp dataset:
```Bash
mkdir yelp_dataset
tar -xf yelp_dataset.tar -C yelp_dataset
```

Prepare embeddings (2021 - train, 2022 - validation and test):
```Bash
sh train_test_split.sh yelp_dataset/yelp_academic_dataset_review.json
sh process_reviews.sh train.json
mv storage/embeddings_chunk_00.hdf5 storage/embeddings_2021.hdf5
sh process_reviews.sh test.json
mv storage/embeddings_chunk_00.hdf5 storage/embeddings_2022.hdf5
```

Users from the validation dataset met in train: 9284 \
Businesses from the validation dataset encountered in train: 17763

Cold start training:
```Bash
python train_cold_start.py --embeddings_train storage/embeddings_2021.hdf5 --train_json train.json --val_json test.json --lr 0.0001 --epochs 250
```
Test ROC-AUC: 0.8133 | Average Precision: 0.8141

Use embeddings from 2022 for training:
```Bash
$ python train.py --embeddings_train storage/embeddings_2021.hdf5 --embeddings_val storage/embeddings_2022.hdf5 --train_json train.json --val_json test.json --lr 0.0001 --epochs 250
```
Test ROC-AUC: 0.8098 | Average Precision: 0.8108
