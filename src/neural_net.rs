use ndarray::{Array, Array2, Dimension};
use ndarray_stats::QuantileExt;
use std::{iter::zip};

use crate::{
    datasets::Dataloader,
    tensor::{
        ops::{Backprop, BinaryOps, ReduceOps, TensorConstructors, UnaryOps},
        Tensor, TensorCore,
    },
};

#[derive(Debug)]
pub struct Model<T: Dataloader> {
    pub w: Vec<Tensor>, // weights
    pub b: Vec<Tensor>, // biases
    pub z: Vec<Tensor>, // sum(w.x + b)
    pub a: Vec<Tensor>, // sigmoid(z)
    dataloader: T,
}

impl<T: Dataloader> Model<T> {
    pub fn init(isize: usize, osize: usize, hidden_layers: Vec<usize>, dataloader: T) -> Model<T> {
        let mut m = Model {
            w: Vec::<Tensor>::new(),
            b: Vec::<Tensor>::new(),
            z: Vec::<Tensor>::new(),
            a: Vec::<Tensor>::new(),
            dataloader,
        };

        let layers = [vec![isize], hidden_layers, vec![osize]].concat();

        for x in 0..layers.len() - 1 {
            m.w.push(TensorCore::randn((layers[x + 1], layers[x])));
            m.b.push(TensorCore::randn((layers[x + 1], 1)));
            m.z.push(TensorCore::zeros((layers[x + 1], 1)));
            m.a.push(TensorCore::zeros((layers[x + 1], 1)));
        }

        return m;
    }

    pub fn train(&mut self, batch_size: u16, epochs: u16) {
        let total_batches = self.dataloader.size() / batch_size;
        for e in 0..epochs {
            for j in 0..total_batches {
                self.train_mini_batch(batch_size.into(), j.into(), 2.);
            }
    
            // Test neural net performance every epoch
            let mut total_correct_pred = 0;
            for i in 0..60_000 {
                let (_x, _y) = self.dataloader.get_by_idx(i);
                let X = TensorCore::new(_x);
                let (Y, _) =  _y.argmax().unwrap();

                let (y, _) = self.feed_forward(&X).data.value.borrow().argmax().unwrap();
                if y == Y {
                    total_correct_pred = total_correct_pred + 1
                }
            }
    
            println!(
                "{} NN perf score {}",
                e,
                (total_correct_pred as f64) * 100. / 60_000.
            );
        }
    }

    pub fn train_mini_batch(&mut self, batch_size: usize, batch_idx: usize, lr: f32) {
        let mut w_grads_batch_aggregate: Vec<Array2<f32>> = (0..self.w.len())
            .map(|l| Array2::zeros(self.w[l].data.value.borrow().dim()))
            .collect();
        let mut b_grads_batch_aggregate: Vec<Array2<f32>> = (0..self.b.len())
            .map(|l| Array2::zeros(self.b[l].data.value.borrow().dim()))
            .collect();

        let batch = self.dataloader.get_batch(batch_size, batch_idx);

        for (x, y) in batch {
            let x_tensor = TensorCore::new(x);
            let y_tensor = TensorCore::new(y);
            self.backprop(&x_tensor, &y_tensor);

            for i in 0..self.w.len() {
                w_grads_batch_aggregate[i] =
                    &w_grads_batch_aggregate[i] + self.w[i].data.grad.borrow().as_ref().unwrap();
                b_grads_batch_aggregate[i] =
                    &b_grads_batch_aggregate[i] + self.b[i].data.grad.borrow().as_ref().unwrap();
            }
        }

        for i in 0..self.w.len() {
            let mut w = self.w[i].data.value.borrow_mut();
            *w = &self.w[i].data.value.borrow() as &Array2<f32> - (lr * (&w_grads_batch_aggregate[i] / batch_size as f32));

            let mut b = self.b[i].data.value.borrow_mut();
            *b = &self.b[i].data.value.borrow() as &Array2<f32> - (lr * (&b_grads_batch_aggregate[i] / batch_size as f32));
        }
    }

    pub fn feed_forward(&self, input: &Tensor) -> Tensor {
        let mut a = Array2::<f32>::zeros((1, 1));
        let mut next_in = &input.data.value.borrow() as &Array2<f32>;
        for (w, b) in zip(&self.w, &self.b) {
            let z = (&w.data.value.borrow() as &Array2<f32>).dot(next_in)
                + &b.data.value.borrow() as &Array2<f32>;
            a = sigmoid_activation(&z);
            next_in = &a;
        }
        return TensorCore::new(a);
    }

    pub fn backprop(&mut self, x: &Tensor, y: &Tensor) {
        // Feed forward
        self.z[0] = x.matmul(&self.w[0]).add(&self.b[0]);
        self.a[0] = self.z[0].sigmoid();

        for i in 1..self.w.len() {
            self.z[i] = self.w[i].matmul(&self.a[i - 1]).add(&self.b[i]);
            self.a[i] = self.z[i].sigmoid();
        }

        let loss = self.a[self.a.len() - 1]
            .sub(y)
            .mul(&TensorCore::fill(
                self.a[self.a.len() - 1].data.value.borrow().dim(),
                0.5,
            ))
            .square()
            .mean();
        loss.backward();
    }
}

fn sigmoid_activation<T: Dimension>(val: &Array<f32, T>) -> Array<f32, T> {
    val.mapv(|val| 1.0 / (1.0 + f32::exp(-val)))
}

