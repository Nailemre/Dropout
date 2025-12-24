import { FC, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { TrainingTabProps } from "./TrainingTab.types";

const TrainingTab: FC<TrainingTabProps> = ({
  modelDropout,
  modelNoDropout,
  trainXs,
  trainYs,
  testXs,
  testYs,
  onTrainingComplete,
}) => {
  const [training, setTraining] = useState(false);
  const [status, setStatus] = useState<string>("");

  const trainModel = async (
    model: tf.Sequential,
    label: string
  ) => {
    setStatus(`Training ${label} model...`);

    await model.fit(trainXs, trainYs, {
      epochs: 15,
      batchSize: 16,
      validationData: [testXs, testYs],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(
            `[${label}] Epoch ${epoch + 1}`,
            logs
          );
          await tf.nextFrame();
        },
      },
    });
  };

  const startTraining = async () => {
    if (training) return;

    setTraining(true);

    // ❗ SIRAYLA eğitiyoruz (çok önemli)
    await trainModel(modelNoDropout, "Without Dropout");
    await trainModel(modelDropout, "With Dropout");

    setTraining(false);
    setStatus("Training complete ✅");
    onTrainingComplete();
  };

  return (
    <div className="border p-4 rounded space-y-3">
      <h2 className="text-xl font-bold">📊 Training</h2>

      <button
        onClick={startTraining}
        disabled={training}
        className="px-4 py-2 bg-blue-600 text-white rounded"
      >
        {training ? "Training..." : "Start Training"}
      </button>

      <div className="text-sm text-gray-600">{status}</div>
    </div>
  );
};

export default TrainingTab;
