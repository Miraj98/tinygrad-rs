pub mod neural_net {
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[derive(Debug)]
    pub struct Layer {
        w: Array2<f64>,
        b: Array2<f64>,
    }

    impl Layer {
        pub fn random_weights(in_size: usize, out_size: usize) -> Layer {
            Layer {
                w: Array2::random((out_size, in_size), Uniform::new(0., 1.)),
                b: Array2::zeros((out_size, 1)),
            }
        }

        pub fn next(&self, prev_layer_out: &Array2<f64>) -> Array2<f64> {
            self.w.dot(prev_layer_out)
        }
    }

    #[derive(Debug)]
    pub struct Model {
        layers: Vec<Layer>,
        out: Array2<f64>,
    }

    impl Model {
        pub fn generate(in_size: usize, out_size: usize, layer_sizes: Vec<usize>) -> Model {
            let mut model = Model {
                layers: Vec::<Layer>::new(),
                out: Array2::zeros((out_size, 1)),
            };

            let n = layer_sizes.len();

            model
                .layers
                .push(Layer::random_weights(in_size, layer_sizes[0]));

            if n > 2 {
                for x in 1..(n - 1) {
                    model
                        .layers
                        .push(Layer::random_weights(layer_sizes[x - 1], layer_sizes[x]));
                }
            }

            model
                .layers
                .push(Layer::random_weights(layer_sizes[n - 1], out_size));

            model
        }

        pub fn forward_pass(&self, input: &Array2<f64>) -> Array2<f64> {
            let mut first_pass = self.layers[0].next(input);

            for x in 1..self.layers.len() {
                first_pass = self.layers[x].next(&first_pass);
            }
            first_pass
        }
    }
}
