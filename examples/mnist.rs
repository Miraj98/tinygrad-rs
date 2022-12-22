use tensor_rs::{dim::Dimension, DataElement};
use tinygrad_rs::{datasets::mnist_data::MnistDataset, layers::{Linear, Layer}, dataloader::Dataset};

fn main() {
    let data = MnistDataset();
}

struct Model(pub Linear<f32>, pub Linear<f32>);

impl Model {
    pub fn new() -> Self {
        Self(Linear::new([784, 256]), Linear::new([256, 10]))
    }

    pub fn train(&mut self, dataset: Dataset<[usize; 3], f32>) {
        for batch in dataset.batch_iter(15) {
            for (x, y) in batch {
                let mut _x = self.0.forward(x);
            }
        }
    }
}
