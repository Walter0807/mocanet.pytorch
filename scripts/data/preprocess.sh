python scripts/data/preprocess.py --data_dir data/mixamo/36_800_24/train -i 32 -s 64 -p 0
python scripts/data/preprocess.py --data_dir data/mixamo/36_800_24/test -i 60 -s 120 -p 0
python scripts/data/rotate_test_set.py --data_dir data/mixamo/36_800_24/test --out_dir data/mixamo/36_800_24/test_random_rotate
python scripts/data/preprocess_solo_dance.py --data_dir data/solo_dance/train -i 32 -s 64 -p 0