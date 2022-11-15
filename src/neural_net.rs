use crate::{
    datasets::Dataloader,
    nn,
    tensor::{
        ops::{binary_ops::BinaryOps, unary_ops::UnaryOps},
        Tensor,
    },
};
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use std::{iter::zip, rc::Rc};

#[derive(Debug)]
pub struct Model<T: Dataloader> {
    pub w: Vec<Rc<Tensor>>, // weights
    pub b: Vec<Rc<Tensor>>, // biases
    _loss_chart: Vec<(f64, f64)>,
    _lr_chart: Vec<(f64, f64)>,
    // pub z: Vec<Tensor>, // sum(w.x + b)
    // pub a: Vec<Tensor>, // sigmoid(z)
    dataloader: T,
}

impl<T: Dataloader> Model<T> {
    pub fn init(isize: usize, osize: usize, hidden_layers: Vec<usize>, dataloader: T) -> Model<T> {
        let mut m = Model {
            w: Vec::<Rc<Tensor>>::new(),
            b: Vec::<Rc<Tensor>>::new(),
            _loss_chart: Vec::<(f64, f64)>::new(),
            _lr_chart: Vec::<(f64, f64)>::new(),
            dataloader,
        };

        let layers = [vec![isize], hidden_layers, vec![osize]].concat();
        for x in 0..layers.len() - 1 {
            m.w.push(Tensor::randn((layers[x + 1], layers[x]), Some(true)));
            m.b.push(Tensor::randn((layers[x + 1], 1), Some(true)));
        }

        return m;
    }

    fn _draw_chart(data: Vec<(f64, f64)>) {
        let root =
            BitMapBackend::new("accuracy.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.margin(10, 10, 10, 10);
        // After this point, we should be able to draw construct a chart context
        let mut chart = ChartBuilder::on(&root)
            // Set the caption of the chart
            .caption("Accuracy plot", ("sans-serif", 40).into_font())
            // Set the size of the label region
            .x_label_area_size(30)
            .y_label_area_size(30)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(0f64..30f64, 75f64..100f64)
            .unwrap();

        chart
            .configure_mesh()
            // We can customize the maximum number of labels allowed for each axis
            .x_labels(30)
            .y_labels(25)
            // We can also change the format of the label text
            .y_label_formatter(&|x| format!("{:.3}", x))
            .draw()
            .unwrap();

        // And we can draw something in the drawing area
        chart.draw_series(LineSeries::new(data, &RED)).unwrap();
        root.present().unwrap();
    }

    pub fn train(&mut self, batch_size: u16, epochs: u16) {
        let total_batches = self.dataloader.size() / batch_size;
        let mut plot_data = vec![(0., 0.); 30];

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
            let total_correct_pred_percent = (total_correct_pred as f64) * 100. / 60_000.;
            plot_data[e as usize] = (e as f64 + 1., total_correct_pred_percent);
            // self._lr_chart.push((e as f64, total_correct_pred_percent));

            println!(
                "{} NN perf score {}",
                e,
                (total_correct_pred as f64) * 100. / 60_000.
            );
        }
        Model::<T>::_draw_chart(plot_data);
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
    }
}
