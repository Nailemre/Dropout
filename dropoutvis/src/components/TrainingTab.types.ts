import * as tf from "@tensorflow/tfjs";

export interface TrainingTabProps {
  modelDropout: tf.Sequential;
  modelNoDropout: tf.Sequential;

  trainXs: tf.Tensor;
  trainYs: tf.Tensor;

  testXs: tf.Tensor;
  testYs: tf.Tensor;

  onTrainingComplete: () => void;
}
