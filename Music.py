import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation

# HyperParameters
lowest_note = midi_manipulation.lowerBound
highest_note = midi_manipulation.upperBound
note_range = highest_note - lowest_note

num_timesteps = 15
n_visible = 2 * note_range * num_timesteps
n_hidden = 50
num_epochs = 200
batch_size = 100
lr = tf.constant(0.005, tf.float32)

x = tf.placeholder(tf.float32, [None, n_visible], name="x")
w = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))

x_sample = gibbs_sample(1)
h = sample(tf.sigmoid(tf.matmul(x, w) + bh))
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, w) + bh))

size_bt = tf.cast(tf.shape(x)[0], tf.float32)
w_adder = tf.mul(lr / size_bt, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.mul(lr / size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
bh_adder = tf.mul(lr / size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0, True))

update = [w.assign_add(w_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song = np.array(song)

        for i in range(1, len(song), batch_size):
            tr_x = song[i:i+batch_size]
            sess.run(update, feed_dict={x: tr_x})

    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i,:]):
            continue
        s = np.reshape(sample[i,:], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(s, "generated_chord_{}".format(i))
