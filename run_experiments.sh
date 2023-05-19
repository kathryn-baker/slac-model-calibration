# first run with the all the different data sets
python experiments/basic_calibration.py --epochs 10 --data_source artifical_data --learning_rate 1e-5 --batch_size 256 --experiment_name test
python experiments/basic_calibration.py --epochs 10 --data_source artifical_data2 --learning_rate 1e-5 --batch_size 256 --experiment_name test
python experiments/basic_calibration.py --epochs 10 --data_source artifical_data3 --learning_rate 1e-5 --batch_size 256 --experiment_name test
python experiments/basic_calibration.py --epochs 10 --data_source artifical_data4 --learning_rate 1e-5 --batch_size 256 --experiment_name test
python experiments/basic_calibration.py --epochs 10 --data_source archive_data --learning_rate 1e-5 --batch_size 256 --experiment_name test
