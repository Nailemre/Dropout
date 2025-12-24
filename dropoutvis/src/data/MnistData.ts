import * as tf from "@tensorflow/tfjs";

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;

const TRAIN_SAMPLES = 1000;
const TEST_SAMPLES = 200;

export class MnistData {
  private images!: Float32Array;
  private labels!: Uint8Array;

  async load() {
    // Load image sprite
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src =
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";

    await new Promise<void>((res) => (img.onload = () => res()));

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

    this.images = new Float32Array(data.length / 4);
    for (let i = 0; i < this.images.length; i++) {
      this.images[i] = data[i * 4] / 255;
    }

    const labelsResp = await fetch(
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"
    );
    this.labels = new Uint8Array(await labelsResp.arrayBuffer());
  }

  getTrainData() {
    const xs = tf.tensor2d(
      this.images.slice(0, TRAIN_SAMPLES * IMAGE_SIZE),
      [TRAIN_SAMPLES, IMAGE_SIZE]
    );

    const ys = tf.oneHot(
      tf.tensor1d(this.labels.slice(0, TRAIN_SAMPLES), "int32"),
      NUM_CLASSES
    );

    return { xs, ys };
  }

  getTestData() {
    const start = TRAIN_SAMPLES * IMAGE_SIZE;

    const xs = tf.tensor2d(
      this.images.slice(start, start + TEST_SAMPLES * IMAGE_SIZE),
      [TEST_SAMPLES, IMAGE_SIZE]
    );

    const ys = tf.oneHot(
      tf.tensor1d(
        this.labels.slice(TRAIN_SAMPLES, TRAIN_SAMPLES + TEST_SAMPLES),
        "int32"
      ),
      NUM_CLASSES
    );

    return { xs, ys };
  }

  /** 🔥 ASIL KRİTİK FONKSİYON */
  getTestExample(index: number) {
    const image = this.images.slice(
      (TRAIN_SAMPLES + index) * IMAGE_SIZE,
      (TRAIN_SAMPLES + index + 1) * IMAGE_SIZE
    );

    const label = this.labels[TRAIN_SAMPLES + index];

    return { image, label };
  }
}
