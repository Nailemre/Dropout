import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import { LocalMnistData } from "../data/LocalMnistData";
import { createModelWithDropout } from "../models/modelWithDropout";
import { createModelWithoutDropout } from "../models/modelWithoutDropout";

import TrainingTab from "../components/TrainingTab";
import PredictionsTab from "../components/PredictionsTab";
import ConfusionMatrixTab from "../components/ConfusionMatrixTab";

const DropoutMnistPage = () => {
  const [data, setData] = useState<LocalMnistData | null>(null);
  const [trained, setTrained] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      const d = new LocalMnistData();
      await d.load();
      setData(d);
    };
    loadData();
  }, []);

  if (!data) {
    return <div>Loading local MNIST data...</div>;
  }

  const { xs: trainXs, ys: trainYs } = data.getTrainData();
  const { xs: testXs, ys: testYs } = data.getTestData();

  const modelDropout = createModelWithDropout();
  const modelNoDropout = createModelWithoutDropout();

  return (
    <div className="p-6 space-y-6">
      <TrainingTab
        modelDropout={modelDropout}
        modelNoDropout={modelNoDropout}
        trainXs={trainXs}
        trainYs={trainYs}
        testXs={testXs}
        testYs={testYs}
        onTrainingComplete={() => setTrained(true)}
      />

      {trained && (
        <>
          <PredictionsTab
            modelDropout={modelDropout}
            modelNoDropout={modelNoDropout}
            data={data}
          />

          <ConfusionMatrixTab
            modelDropout={modelDropout}
            modelNoDropout={modelNoDropout}
            testXs={testXs}
            testYs={testYs}
          />
        </>
      )}
    </div>
  );
};

export default DropoutMnistPage;
