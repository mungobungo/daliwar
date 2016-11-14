# daliwar
salvador dali + andy warhol


building runner(single file) given model
```

daliwar/model_runner $ docker build -t daliwar/modelrunner-0.9 .

daliwar/model_runner $ docker run -v /home/oleg/projects/daliwar/model_runner/images:/images
    \  -e IMAGE=/images/14021430525_e06baf93a9.jpg
    -t -i daliwar/modelrunner-0.9
```

output

```
filename : /images/14021430525_e06baf93a9.jpg
W tensorflow/core/framework/op_def_util.cc:332] Op BatchNormWithGlobalNormalization  is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
daisy (score = 0.55027)
sunflowers (score = 0.36755)
tulips (score = 0.04926)
dandelion (score = 0.03033)
roses (score = 0.00260)
```