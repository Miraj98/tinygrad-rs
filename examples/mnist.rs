use std::time::Instant;

use tensor_rs::{impl_unary_ops::TensorUnaryOps, prelude::BackwardOps};
use tinygrad_rs::{
    dataloader::Dataset,
    datasets::mnist_data::MnistDataset,
    layers::{Layer, Linear},
    loss::CrossEntropyLoss,
};

fn main() {
    let data = MnistDataset();
    let mut model = Model::new();
    model.train(data);
}

struct Model(pub Linear<f32>, pub Linear<f32>);

impl Model {
    pub fn new() -> Self {
        Self(Linear::new(256, 784), Linear::new(10, 256))
    }

    pub fn train(&mut self, dataset: Dataset<[usize; 3], f32>) {
        let criterion = CrossEntropyLoss();
        let batch_size = 15;
        let alpha = 3. / batch_size as f32;
        let epochs = 30;
        for e in 0..epochs {
            let start = Instant::now();
            for batch in dataset.batch_iter(batch_size) {
                let (mut wg0, mut bg0) = self.0.zeros();
                let (mut wg1, mut bg1) = self.1.zeros();
                for (x, y) in batch {
                    let mut a = self.0.forward(x);
                    a = self.1.forward(a.sigmoid()).sigmoid();
                    let loss = criterion(a, y);
                    let grads = loss.backward();
                    wg0 += grads.grad(self.0.weight());
                    bg0 += grads.grad(self.0.bias());
                    wg1 += grads.grad(self.1.weight());
                    bg1 += grads.grad(self.1.bias());
                    self.0.weight().put_backward_ops(Some(BackwardOps::new()));
                    self.1.weight().put_backward_ops(Some(BackwardOps::new()));
                    self.0.bias().put_backward_ops(Some(BackwardOps::new()));
                    self.1.bias().put_backward_ops(Some(BackwardOps::new()));
                }
                wg0 *= alpha;
                bg0 *= alpha;
                wg1 *= alpha;
                bg1 *= alpha;

                *self.0.weight_mut() -= wg0;
                *self.0.bias_mut() -= bg0;
                *self.1.weight_mut() -= wg1;
                *self.1.bias_mut() -= bg1;
            }
            println!(
                "Completed epoch {e} in {} secs",
                start.elapsed().as_secs_f64()
            );
        }
    }
}
