use crate::{
    datasets::Dataloader,
    tensor::{
        ops::{binary_ops::BinaryOps, reduce_ops::ReduceOps, unary_ops::UnaryOps},
        Tensor,
    }, nn,
};
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use std::{iter::zip, rc::Rc};

#[derive(Debug)]
pub struct Model<T: Dataloader> {
    pub w: Vec<Rc<Tensor>>, // weights
    pub b: Vec<Rc<Tensor>>, // biases
    // pub z: Vec<Tensor>, // sum(w.x + b)
    // pub a: Vec<Tensor>, // sigmoid(z)
    dataloader: T,
}

impl<T: Dataloader> Model<T> {
    pub fn init(isize: usize, osize: usize, hidden_layers: Vec<usize>, dataloader: T) -> Model<T> {
        let mut m = Model {
            w: Vec::<Rc<Tensor>>::new(),
            b: Vec::<Rc<Tensor>>::new(),
            dataloader,
        };

        let layers = [vec![isize], hidden_layers, vec![osize]].concat();
        for x in 0..layers.len() - 1 {
            m.w.push(Tensor::randn((layers[x + 1], layers[x]), Some(true)));
            m.b.push(Tensor::randn((layers[x + 1], 1), Some(true)));
        }

        return m;
    }

    pub fn train(&mut self, batch_size: u16, epochs: u16) {
        let total_batches = self.dataloader.size() / batch_size;
        println!("Total batches {}", total_batches);
        for e in 0..epochs {
            for j in 0..total_batches {
                self.train_mini_batch(batch_size.into(), j.into(), 2.);
            }

            // Test neural net performance every epoch
            let mut total_correct_pred = 0;
            for i in 0..60_000 {
                let (x, y) = self.dataloader.get_by_idx(i);
                let (y_arg, _) = y.argmax().unwrap();
                let (y_pred, _) = self.feed_forward(&x).ndarray().argmax().unwrap();
                if y_arg == y_pred {
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

    pub fn train_mini_batch(&mut self, batch_size: usize, batch_idx: usize, lr: f64) {
        let mut w_grad_agg: Vec<Array2<f64>> =
            self.w.iter().map(|w| Array2::zeros(w.dim())).collect();
        let mut b_grad_agg: Vec<Array2<f64>> =
            self.b.iter().map(|b| Array2::zeros(b.dim())).collect();
        let batch = self.dataloader.get_batch(batch_size, batch_idx);

        for (x, y) in batch {
            let loss = self.backprop(&x, &y);
            for i in 0..self.w.len() {
                w_grad_agg[i] = &w_grad_agg[i] + self.w[i].grad().as_ref().unwrap();
                b_grad_agg[i] = &b_grad_agg[i] + self.b[i].grad().as_ref().unwrap();
            }
            loss.zero_grad();
        }

        let mut _w: Vec<Array2<f64>> = self.w.iter().map(|v| Array2::zeros(v.dim())).collect();
        let mut _b: Vec<Array2<f64>> = self.b.iter().map(|v| Array2::zeros(v.dim())).collect();

        for i in 0..self.w.len() {
            let w = &self.w[i].ndarray() as &Array2<f64>;
            let del_w = (lr / batch_size as f64) * &w_grad_agg[i];
            _w[i] = w - del_w;

            let b = &self.b[i].ndarray() as &Array2<f64>;
            let del_b = (lr / batch_size as f64) * &b_grad_agg[i];
            _b[i] = b - del_b;
        }

        for i in 0..self.w.len() {
            self.w[i] = Tensor::from(&_w[i], Some(true));
            self.b[i] = Tensor::from(&_b[i], Some(true));
        }
    }

    pub fn feed_forward(&self, input: &Array2<f64>) -> Rc<Tensor> {
        let input_tensor = Tensor::from(input, Some(false));
        let mut a = Tensor::zeros((1, 1), Some(false));
        let mut next_in = &input_tensor;

        for (w, b) in zip(&self.w, &self.b) {
            a = w.matmul(next_in).add(b).sigmoid();
            next_in = &a;
        }
        return a;
    }

    pub fn backprop(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Rc<Tensor> {
        let xt = Tensor::from(x, Some(true));
        let yt = Tensor::from(y, Some(true));

        // Feed forward
        let mut a = self.w[0].matmul(&xt).add(&self.b[0]).sigmoid();
        for i in 1..self.w.len() {
            a = self.w[i].matmul(&a).add(&self.b[i]).sigmoid();
        }

        // Find loss and call backward on it
        let loss = nn::cross_entropy(&a, &yt);
        loss.backward();
        loss


        // let loss = a.sub(&yt).mul_scalar(0.5).square().mean();
        // loss.backward();
        // loss
    }
}
