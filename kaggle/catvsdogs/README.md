screenshot from kaggle with competition
download train images, download test images, download example result for submission
(screenshot with folder containing 3 images)
unzip training model
25K files would take a while

copy cats and dogs to appropriate folders

```
ls ~/projects/dogsvscats/train | tail -n 10
dog.9993.jpg
dog.9994.jpg
dog.9995.jpg
dog.9996.jpg
dog.9997.jpg
dog.9998.jpg
dog.9999.jpg
dog.999.jpg
dog.99.jpg
dog.9.jpg
```

```
ls ~/projects/dogsvscats/train | head -n 10
cat.0.jpg
cat.10000.jpg
cat.10001.jpg
cat.10002.jpg
cat.10003.jpg
cat.10004.jpg
cat.10005.jpg
cat.10006.jpg
cat.10007.jpg
cat.10008.jpg
```

```
cd ~/projects/dogsvscats/train 

mkdir cat && mv cat*.jpg cat/
mkdir dog && mv dog*.jpg dog/ 

```

now we need to run retraining docker
```
docker run -it -v $HOME/projects/dogsvscats/:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel

cd /tensorflow

time python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 1000 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/train
```

now you need wait until stuff is trained. it might take some time


building runner(single file) given model
```

daliwar/model_runner $ docker build -t daliwar/catdog-0.3 .

daliwar/model_runner $  docker run -v /home/oleg/projects/dogsvscats:/data     -t -i daliwar/catdog-0.3
```