import * as tf from "@tensorflow/tfjs";

const IMAGE_SIZE = 28;
const NUM_CLASSES = 10;

async function loadImage(path: string): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = path;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = IMAGE_SIZE;
      canvas.height = IMAGE_SIZE;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      const { data } = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);

      const pixels = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
      for (let i = 0; i < pixels.length; i++) {
        pixels[i] = data[i * 4] / 255;
      }
      resolve(pixels);
    };
    img.onerror = reject;
  });
}

export class LocalMnistData {
  trainImages: number[][] = [];
  trainLabels: number[] = [];
  testImages: number[][] = [];
  testLabels: number[] = [];

  async load() {
    await this.loadSplit("train");
    await this.loadSplit("test");
  }

  private async loadSplit(split: "train" | "test") {
    for (let label = 0; label < NUM_CLASSES; label++) {
      const maxImages = split === "train" ? 30 : 5;

      for (let i = 0; i < maxImages; i++) {
        const path = `/mnist/${split}/${label}/${i}.png`;

        try {
          const img = await loadImage(path);

          if (split === "train") {
            this.trainImages.push(Array.from(img));
            this.trainLabels.push(label);
          } else {
            this.testImages.push(Array.from(img));
            this.testLabels.push(label);
          }
        } catch {
          console.warn(`Eksik dosya: ${path}`);
        }
      }
    }
  }

  getTrainData() {
    return {
      xs: tf.tensor2d(this.trainImages),
      ys: tf.oneHot(
        tf.tensor1d(this.trainLabels, "int32"),
        NUM_CLASSES
      ),
    };
  }

  getTestData() {
    return {
      xs: tf.tensor2d(this.testImages),
      ys: tf.oneHot(
        tf.tensor1d(this.testLabels, "int32"),
        NUM_CLASSES
      ),
    };
  }

  /** 🔍 Prediction için %100 doğru örnek */
  getTestExample(index: number) {
    return {
      image: this.testImages[index],
      label: this.testLabels[index],
    };
  }
}
