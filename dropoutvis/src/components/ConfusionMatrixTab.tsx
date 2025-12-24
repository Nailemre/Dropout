import * as tf from "@tensorflow/tfjs";

export interface ConfusionMatrixTabProps {
  modelDropout: tf.Sequential;
  modelNoDropout: tf.Sequential;

  testXs: tf.Tensor;
  testYs: tf.Tensor;
}

export default function ConfusionMatrixTab({
  modelDropout,
  modelNoDropout,
  testXs,
  testYs,
}: ConfusionMatrixTabProps) {
  const preds = modelDropout.predict(testXs) as tf.Tensor2D;

  const predLabels = preds.argMax(1) as tf.Tensor1D;
  const trueLabels = testYs.argMax(1) as tf.Tensor1D;

  const matrix = tf
    .math.confusionMatrix(trueLabels, predLabels, 10)
    .arraySync() as number[][];

  return (
    <div>
      <h2 className="text-xl font-bold">📋 Confusion Matrix</h2>
      {matrix.map((row, i) => (
        <div key={i} className="flex">
          {row.map((v, j) => (
            <div
              key={j}
              className={`w-8 h-8 flex items-center justify-center text-xs
                ${i === j ? "bg-green-500" : v > 0 ? "bg-red-400" : "bg-gray-100"}`}
            >
              {v}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
