use tinygrad_rust::{neural_net, datasets::mnist};
use ndarray::Array2;
use ndarray_stats::QuantileExt;

fn main() {
    let mnist_data = mnist::load_data();
    let batch_size = 5;
    let epochs = 30;
    let mut model = neural_net::Model::init(28 * 28, 10, vec![30]);

    for e in 0..epochs {
        for j in 0..(mnist_data.DATA_SET_SIZE / batch_size) {
            let train_in_batch: Vec<Array2<f64>> = (0..batch_size)
                .map(|val| mnist_data.get_image_nn_input((j * batch_size + val) as usize))
                .collect();
            let train_out_batch = (0..batch_size)
                .map(|val| mnist_data.get_image_label_vector((j * batch_size + val) as usize))
                .collect();
            model.train_mini_batch(&train_in_batch, &train_out_batch, 2.);
        }

        // Test neural net performance every epoch
        let mut total_correct_pred = 0;
        for i in 0..60_000 {
            let in_vec = mnist_data.get_image_nn_input(i);
            let out = mnist_data.get_image_label(i);
            let (m, _n) = model.feed_forward(&in_vec).argmax().unwrap();
            if m as u8 == out {
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