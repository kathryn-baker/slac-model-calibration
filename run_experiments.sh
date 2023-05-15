# first run with the artificial data
python experiments/basic_calibration.py --epochs 10000 --data_source artifical_data --learning_rate 1e-6
python experiments/individual_outputs.py --epochs 10000 --data_source artifical_data --learning_rate 1e-6
python experiments/linear_layer.py --epochs 20000 --data_source artifical_data --learning_rate 1e-6 --activation tanh
python experiments/linear_layer.py --epochs 20000 --data_source artifical_data --learning_rate 1e-6 --activation none
python experiments/linear_layer.py --epochs 20000 --data_source artifical_data --learning_rate 1e-6 --activation relu

# then run with the second set of artificial data
python experiments/basic_calibration.py --epochs 10000 --data_source artificial_data2 --learning_rate 1e-6
python experiments/individual_outputs.py --epochs 10000 --data_source artificial_data2 --learning_rate 1e-6
python experiments/linear_layer.py --epochs 20000 --data_source artificial_data2 --learning_rate 1e-6 --activation tanh
python experiments/linear_layer.py --epochs 20000 --data_source artificial_data2 --learning_rate 1e-6 --activation none
python experiments/linear_layer.py --epochs 20000 --data_source artificial_data2 --learning_rate 1e-6 --activation relu

# finally run with real data
python experiments/basic_calibration.py --epochs 10000 --data_source archive_data --learning_rate 1e-6
python experiments/individual_outputs.py --epochs 10000 --data_source archive_data --learning_rate 1e-6
python experiments/linear_layer.py --epochs 20000 --data_source archive_data --learning_rate 1e-6 --activation tanh
python experiments/linear_layer.py --epochs 20000 --data_source archive_data --learning_rate 1e-6 --activation none
python experiments/linear_layer.py --epochs 20000 --data_source archive_data --learning_rate 1e-6 --activation relu