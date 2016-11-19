import time
import sys
import tensorflow as tf
import os
import glob
import pandas as  pd
import numpy as np

DATA_FOLDER='/data/'
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph2.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.Session()
    # Feed the image_data as input to the graph and get first prediction
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
print("loaded tensor")

test_files = glob.glob(DATA_FOLDER + '/test/*.jpg')

submit_results = []
index = 0
for image_path in test_files:
    print("iteration : %d" % index)
    index = index + 1
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print(image_path)
    basename = os.path.basename(image_path) 
    print basename
    print(predictions)
    print("dog : %s " % (predictions[0][0] > 0.5))
    submit_label = (round(predictions[0][0]))
    print("dog : %f " % submit_label)
    submit_id = int(basename.split('.')[0])
    submit_results.append( (submit_id, submit_label))
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

submit_file = DATA_FOLDER + '/' + time.strftime("%Y-%m-%d-%H-%M-%S") + '-kaggle_submit.csv'

print("starting saving dataset")

df = pd.DataFrame(data=submit_results, columns = ['id','label'])
res = df.sort_values(by=['id'])
res.to_csv(submit_file,index=False,header=True)
print("done")
