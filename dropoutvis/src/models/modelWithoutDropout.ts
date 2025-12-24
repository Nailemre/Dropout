import * as tf from '@tensorflow/tfjs';

export function createModelWithoutDropout(): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 128,
      activation: 'relu',
      inputShape: [784],
    })
  );

  model.add(
    tf.layers.dense({
      units: 10,
      activation: 'softmax',
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}
