pub mod neural_net {
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn get_immut_ref<T>(a: &mut T) -> &T {
        &*a
    }

    #[derive(Debug)]
    pub struct Model {
        pub w: Vec<Array2<f64>>, // weights
        pub b: Vec<Array2<f64>>, // biases
        w_grad: Vec<Array2<f64>>,
        b_grad: Vec<Array2<f64>>,
        error_rates: Vec<Array2<f64>>,
        a: Vec<Array2<f64>>,  // final activations
        da: Vec<Array2<f64>>, // derivative of activations
        E: Array2<f64>,       // elementwise error; E = (y - a)
        cost_val: f64,        // E / N
    }

    impl Model {
        pub fn init(isize: usize, osize: usize, hidden_layers: Vec<usize>) -> Model {
            let mut m = Model {
                w: Vec::<Array2<f64>>::new(),
                b: Vec::<Array2<f64>>::new(),
                w_grad: Vec::<Array2<f64>>::new(),
                b_grad: Vec::<Array2<f64>>::new(),
                a: Vec::<Array2<f64>>::new(),
                da: Vec::<Array2<f64>>::new(),
                error_rates: Vec::<Array2<f64>>::new(),
                E: Array2::<f64>::zeros((osize, 1)),
                cost_val: -1.,
            };

            m.w.push(Array2::random(
                (hidden_layers[0], isize),
                Uniform::new(-1.0, 1.0),
            ));
            m.w_grad.push(Array2::zeros((hidden_layers[0], isize)));
            m.b_grad.push(Array2::zeros((hidden_layers[0], 1)));
            m.error_rates.push(Array2::zeros((hidden_layers[0], 1)));
            m.b.push(Array2::zeros((hidden_layers[0], 1)));
            m.a.push(Array2::zeros((hidden_layers[0], 1)));
            m.da.push(Array2::zeros((hidden_layers[0], 1)));
            for x in 0..hidden_layers.len() - 1 {
                m.w.push(Array2::random(
                    (hidden_layers[x + 1], hidden_layers[x]),
                    Uniform::new(-1., 1.),
                ));
                m.b.push(Array2::zeros((hidden_layers[x + 1], 1)));
                m.a.push(Array2::zeros((hidden_layers[x + 1], 1)));
                m.da.push(Array2::zeros((hidden_layers[x + 1], 1)));
                m.w_grad
                    .push(Array2::zeros((hidden_layers[x + 1], hidden_layers[x])));
                m.b_grad.push(Array2::zeros((hidden_layers[x + 1], 1)));
                m.error_rates.push(Array2::zeros((hidden_layers[x + 1], 1)));
            }
            m.w.push(Array2::random(
                (osize, hidden_layers[hidden_layers.len() - 1]),
                Uniform::new(-1., 1.),
            ));
            m.b.push(Array2::zeros((osize, 1)));
            m.a.push(Array2::zeros((osize, 1)));
            m.da.push(Array2::zeros((osize, 1)));
            m.w_grad.push(Array2::zeros((
                osize,
                hidden_layers[hidden_layers.len() - 1],
            )));
            m.b_grad.push(Array2::zeros((osize, 1)));
            m.error_rates.push(Array2::zeros((osize, 1)));

            return m;
        }

        pub fn train(&mut self, input: &Array2<f64>, output: &Array2<f64>, lr: f64) {
            let mut keep_training = true;
            let mut prev_cost = self.cost_val;
            let max_iters = 100_000;
            let mut iter = 0;

            while keep_training && iter < max_iters {
                self.feed_forward(input);
                self.cost(output);
                self.calculate_error_rates(output);
                self.update_weights(input, -lr);

                if self.cost_val > prev_cost && prev_cost >= 0. {
                    keep_training = false;
                }

                prev_cost = self.cost_val;
                iter = iter + 1;
            }

            self.feed_forward(input);
        }

        pub fn feed_forward(&mut self, input: &Array2<f64>) {
            self.a[0] = (&self.w[0].dot(input) + &self.b[0]).map(sigmoid_activation);
            self.da[0] = self.a[0].map(sigmoid_derivative);
            for i in 1..self.a.len() {
                self.a[i] = (&self.w[i].dot(&self.a[i - 1]) + &self.b[i]).map(sigmoid_activation);
                self.da[i] = self.a[i].map(sigmoid_derivative);
            }
        }

        fn calculate_error_rates(&mut self, expected_out: &Array2<f64>) {
            let diff = self.get_output_vec() - expected_out;
            let layers_len = get_immut_ref(self).a.len();
            self.error_rates[layers_len - 1] = diff * &self.da[layers_len - 1];

            for l in (0..layers_len - 1).rev() {
                let next_layer_len = get_immut_ref(self).a[l + 1].len();
                let curr_layer_len = get_immut_ref(self).a[l].len();
                let w = &self.w[l + 1];

                self.error_rates[l] = Array2::<f64>::zeros([curr_layer_len, 1]);

                for next in 0..next_layer_len {
                    for curr in 0..w.shape()[1] {
                        self.error_rates[l][(curr, 0)] = self.error_rates[l][(curr, 0)]
                            + (self.w[l + 1][(next, curr)]
                                * self.error_rates[l + 1][(next, 0)]
                                * self.a[l][(curr, 0)]
                                * (1. - self.a[l][(curr, 0)]));
                    }
                }
            }
        }

        fn update_weights(&mut self, input: &Array2<f64>, lr: f64) {
            let total_layers = get_immut_ref(self).a.len();

            // For layer idx == 0
            let layer_size = get_immut_ref(self).a[0].len();
            for j in 0..layer_size {
                for k in 0..input.len() {
                    self.w_grad[0][(j, k)] = self.error_rates[0][(j, 0)] * input[(k, 0)];
                }
            }

            for l in 1..total_layers {
                let layer_size = get_immut_ref(self).a[l].len();
                for j in 0..layer_size {
                    let prev_layer_size = get_immut_ref(self).a[l - 1].len();
                    for k in 0..prev_layer_size {
                        self.w_grad[l][(j, k)] =
                            self.error_rates[l][(j, 0)] * self.a[l - 1][(k, 0)];
                    }
                }
            }

            for i in 0..total_layers {
                self.w[i] = &self.w[i] + (lr * &self.w_grad[i]);
            }
        }

        pub fn get_output_vec(&self) -> &Array2<f64> {
            &self.a[self.a.len() - 1]
        }

        fn cost(&mut self, ideal_out: &Array2<f64>) {
            let mut sum = 0.;
            self.E = &self.a[self.a.len() - 1] - ideal_out;
            for i in 0..self.E.len() {
                sum = sum + self.E[(i, 0)] * self.E[(i, 0)];
                self.E[(i, 0)] = self.E[(i, 0)] * self.E[(i, 0)] / 2.0;
            }
            self.cost_val = sum / (ideal_out.len() as f64);
        }
    }

    fn sigmoid_activation(val: &f64) -> f64 {
        1.0 / (1.0 + f64::exp(-val))
    }

    fn sigmoid_derivative(sigmoid_val: &f64) -> f64 {
        sigmoid_val * (1.0 - sigmoid_val)
    }

    // impl Model {
    //     fn cost_derivative(&self, y: Array2<f64>) -> Array2<f64> {
    //         (y - &self.out).mapv(|val| val * 2.)
    //     }
    // }
}
