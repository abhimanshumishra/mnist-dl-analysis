# Classifying Fashion-MNIST images using Deep Neural Networks

## Setup
git clone https://github.com/abhimanshumishra/ml-project.git
Go to the release-data folder and uncompress the .feats.npy.zip files and delete the zip files after that.
conda create --name ml-project python=3.7
conda activate ml-project
For Mac OS X,
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
For Linux or Windows:
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
pip install -r requirements.txt

## Main - Training and Testing Best Models
python main.py train --data-dir release-data --model-save ff.torch --model simple-ff --train-steps 1500 --log-file best-ff-logs.csv --learning-rate 0.001
python main.py predict --data-dir release-data --model-save ff.torch --predictions-file ff-preds.txt
python main.py train --data-dir release-data --model-save cnn.torch --model simple-cnn --train-steps 2500 --log-file best-cnn-logs.csv --cnn-n1-channels 128
python main.py predict --data-dir release-data --model-save cnn.torch --predictions-file cnn-preds.txt
python main.py train --data-dir release-data --model-save best.torch --model best --train-steps 2500 --log-file best-hybrid-logs.csv
python main.py predict --data-dir release-data --model-save best.torch --predictions-file best-preds.txt

## Sweeping - Training Models for Hyperparameter Sweeps
python main.py train --data-dir release-data --model-save ff0.torch --model simple-ff --train-steps 1500 --log-file ff0-logs.csv --learning-rate 0.000001
python main.py train --data-dir release-data --model-save ff1.torch --model simple-ff --train-steps 1500 --log-file ff1-logs.csv --learning-rate 0.00001
python main.py train --data-dir release-data --model-save ff2.torch --model simple-ff --train-steps 1500 --log-file ff2-logs.csv --learning-rate 0.0001
python main.py train --data-dir release-data --model-save ff3.torch --model simple-ff --train-steps 1500 --log-file ff3-logs.csv --learning-rate 0.001
python main.py train --data-dir release-data --model-save ff4.torch --model simple-ff --train-steps 1500 --log-file ff4-logs.csv --learning-rate 0.01
python main.py train --data-dir release-data --model-save ff5.torch --model simple-ff --train-steps 1500 --log-file ff5-logs.csv --learning-rate 0.1
python main.py train --data-dir release-data --model-save ff6.torch --model simple-ff --train-steps 1500 --log-file ff6-logs.csv --learning-rate 1

python main.py train --data-dir release-data --model-save cnn0.torch --model simple-cnn --train-steps 1500 --log-file cnn0-logs.csv --cnn-n1-channels 2
python main.py train --data-dir release-data --model-save cnn1.torch --model simple-cnn --train-steps 1500 --log-file cnn1-logs.csv --cnn-n1-channels 4
python main.py train --data-dir release-data --model-save cnn2.torch --model simple-cnn --train-steps 1500 --log-file cnn2-logs.csv --cnn-n1-channels 8
python main.py train --data-dir release-data --model-save cnn3.torch --model simple-cnn --train-steps 1500 --log-file cnn3-logs.csv --cnn-n1-channels 16
python main.py train --data-dir release-data --model-save cnn4.torch --model simple-cnn --train-steps 1500 --log-file cnn4-logs.csv --cnn-n1-channels 32
python main.py train --data-dir release-data --model-save cnn5.torch --model simple-cnn --train-steps 1500 --log-file cnn5-logs.csv --cnn-n1-channels 40
python main.py train --data-dir release-data --model-save cnn6.torch --model simple-cnn --train-steps 1500 --log-file cnn6-logs.csv --cnn-n1-channels 64
python main.py train --data-dir release-data --model-save cnn7.torch --model simple-cnn --train-steps 1500 --log-file cnn7-logs.csv --cnn-n1-channels 128

python main.py train --data-dir release-data --model-save best00.torch --model best --train-steps 1500 --log-file best00-logs.csv --best-lin1-trans 80
python main.py train --data-dir release-data --model-save best01.torch --model best --train-steps 1500 --log-file best01-logs.csv --best-lin1-trans 100
python main.py train --data-dir release-data --model-save best02.torch --model best --train-steps 1500 --log-file best02-logs.csv --best-lin1-trans 120
python main.py train --data-dir release-data --model-save best03.torch --model best --train-steps 1500 --log-file best03-logs.csv --best-lin1-trans 144
python main.py train --data-dir release-data --model-save best04.torch --model best --train-steps 1500 --log-file best04-logs.csv --best-lin1-trans 192

python main.py train --data-dir release-data --model-save best10.torch --model best --train-steps 1500 --log-file best10-logs.csv --best-n1-channels 3
python main.py train --data-dir release-data --model-save best11.torch --model best --train-steps 1500 --log-file best11-logs.csv --best-n1-channels 4
python main.py train --data-dir release-data --model-save best12.torch --model best --train-steps 1500 --log-file best12-logs.csv --best-n1-channels 5
python main.py train --data-dir release-data --model-save best13.torch --model best --train-steps 1500 --log-file best13-logs.csv --best-n1-channels 2

