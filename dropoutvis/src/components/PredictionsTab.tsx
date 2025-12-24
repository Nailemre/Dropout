import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";
import { LocalMnistData } from "../data/LocalMnistData";

interface PredictionsTabProps {
  modelDropout: tf.Sequential;
  modelNoDropout: tf.Sequential;
  data: LocalMnistData;
}

interface PredictionResult {
  trueLabel: number;
  image: number[];
  dropLabel: number;
  dropConf: number;
  noDropLabel: number;
  noDropConf: number;
}

export default function PredictionsTab({
  modelDropout,
  modelNoDropout,
  data,
}: PredictionsTabProps) {
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const [results, setResults] = useState<PredictionResult[]>([]);

  const SAMPLE_COUNT = 12;

  // 🔥 SADECE DATA HAZIR OLUNCA ÇALIŞIR
  useEffect(() => {
    if (!data || data.testImages.length === 0) return;

    const count = Math.min(SAMPLE_COUNT, data.testImages.length);
    const newResults: PredictionResult[] = [];

    for (let i = 0; i < count; i++) {
      const sample = data.getTestExample(i);
      if (!sample) continue;

      const { image, label } = sample;

      const drop = predict(image, modelDropout);
      const noDrop = predict(image, modelNoDropout);

      newResults.push({
        image,
        trueLabel: label,
        dropLabel: drop.label,
        dropConf: drop.conf,
        noDropLabel: noDrop.label,
        noDropConf: noDrop.conf,
      });
    }

    setResults(newResults);
  }, [data, modelDropout, modelNoDropout]);

  // 🎯 TEK BİR TAHMİN
  const predict = (image: number[], model: tf.Sequential) => {
    const input = tf.tensor2d([image], [1, 784]);
    const preds = model.predict(input) as tf.Tensor2D;

    const label = (preds.argMax(1).arraySync() as number[])[0];
    const conf = (preds.max(1).arraySync() as number[])[0];

    tf.dispose([input, preds]);
    return { label, conf };
  };

  // 🖼️ CANVAS ÇİZİMİ
  useEffect(() => {
    results.forEach((res, i) => {
      const canvas = canvasRefs.current[i];
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const imgData = ctx.createImageData(28, 28);
      for (let j = 0; j < res.image.length; j++) {
        const p = res.image[j] * 255;
        imgData.data[j * 4] = p;
        imgData.data[j * 4 + 1] = p;
        imgData.data[j * 4 + 2] = p;
        imgData.data[j * 4 + 3] = 255;
      }

      ctx.putImageData(imgData, 0, 0);
    });
  }, [results]);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">🎯 Predictions (Local MNIST)</h2>

      {results.map((res, i) => (
        <div key={i} className="flex gap-6 border p-4 rounded">
          <canvas
    ref={(el) => {
  canvasRefs.current[i] = el;
}}
            width={28}
            height={28}
            className="border"
          />

          <div>
            <div className="text-xs text-gray-500">True</div>
            <div className="text-xl font-bold">{res.trueLabel}</div>
          </div>

          <div>
            <div className="text-xs text-gray-500">With Dropout</div>
            <div
              className={`text-xl font-bold ${
                res.dropLabel === res.trueLabel
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              {res.dropLabel}
            </div>
            <div className="text-xs">
              Conf: {(res.dropConf * 100).toFixed(1)}%
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-500">Without Dropout</div>
            <div
              className={`text-xl font-bold ${
                res.noDropLabel === res.trueLabel
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              {res.noDropLabel}
            </div>
            <div className="text-xs">
              Conf: {(res.noDropConf * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
