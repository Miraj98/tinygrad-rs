use std::time::Instant;

use tensor_rs::{impl_unary_ops::TensorUnaryOps, prelude::BackwardOps};
use tinygrad_rs::{
    dataloader::Dataset,
    datasets::mnist_data::MnistDataset,
    layers::{Layer, Linear, Conv2d},
    loss::CrossEntropyLoss,
};

fn main() {
    let data = MnistDataset();
    let mut model = Model::new();
    model.train(data);
}

struct Model(pub Conv2d<f32>,  pub Linear<f32>, pub Linear<f32>);

impl Model {
    pub fn new() -> Self {
        Self(Conv2d::new(1, 21, (5, 5), (1, 1)), Linear::new(30, 784), Linear::new(10, 30))
    }

    pub fn train(&mut self, dataset: Dataset<[usize; 3], f32>) {
        let criterion = CrossEntropyLoss();
        let batch_size = 15;
        let alpha = 3. / batch_size as f32;
        let epochs = 30;
        for e in 0..epochs {
            println!("Epoch {}", e);
            let start = Instant::now();
            for batch in dataset.batch_iter(batch_size) {
                // let batch_start = Instant::now();
                let (mut wg0, mut bg0) = self.0.zeros();
                let (mut wg1, mut bg1) = self.1.zeros();
                let (mut wg2, mut bg2) = self.2.zeros();
                for (i, (x, y)) in batch.enumerate() {
                    let mut a = self.0.forward(x).sigmoid();
                    a = self.1.forward(a).sigmoid();
                    a = self.2.forward(a).sigmoid();
                    let loss = criterion(a, y);
                    let grads = loss.backward();
                    wg0 += grads.grad(self.0.weight());
                    // bg0 += grads.grad(self.0.bias());
                    wg1 += grads.grad(self.1.weight());
                    bg1 += grads.grad(self.1.bias());
                    wg2 += grads.grad(self.2.weight());
                    bg2 += grads.grad(self.2.bias());
                    self.0.weight_mut().put_backward_ops(Some(BackwardOps::new()));
                }
                wg0 *= alpha;
                // bg0 *= alpha;
                wg1 *= alpha;
                bg1 *= alpha;
                wg2 *= alpha;
                bg2 *= alpha;

                *self.0.weight_mut() -= wg0;
                *self.1.weight_mut() -= wg1;
                *self.1.bias_mut() -= bg1;
                *self.2.weight_mut() -= wg2;
                *self.2.bias_mut() -= bg2;
                // println!("Batch time {}", batch_start.elapsed().as_secs_f64());
            }
            println!(
                "Completed epoch {e} in {} secs",
                start.elapsed().as_secs_f64()
            );
        }
    }
}

