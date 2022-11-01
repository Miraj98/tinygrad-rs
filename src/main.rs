mod mnist;
mod tinygrad;

use mnist::MnistTrainData;
use ndarray::Array2;
use tinygrad::nn::neural_net;

fn main() {
    let mnist_data = MnistTrainData::load();
    let idx = 0;
    let batch_size = 10;

    let mut model = neural_net::Model::init(28 * 28, 10, vec![30]);
    for i in 0..(mnist_data.DATA_SET_SIZE / batch_size) {
        let train_in_batch: Vec<Array2<f64>> = (0..batch_size)
            .map(|val| mnist_data.get_image_nn_input((i * batch_size + val) as usize))
            .collect();
        let train_out_batch = (0..batch_size)
            .map(|val| mnist_data.get_image_label_vector((i * batch_size + val) as usize))
            .collect();
        model.train_mini_batch(&train_in_batch, &train_out_batch, 30, 3.);

    }

    println!("\n\n\n\n\n\n\n\n\n");
    for i in 0..20 {
        let in_vec = mnist_data.get_image_nn_input(i);
        let out_vec = mnist_data.get_image_label_vector(i);
        let output_vec = model.feed_forward(&in_vec);
        println!("Output\n{:?}\n\n Input\n{:?}\n\n\n\n\n", output_vec, out_vec);
    }

    println!("{:?} {:?}", mnist_data.get_image_label_vector(idx), mnist_data.get_image_label(idx));
}
