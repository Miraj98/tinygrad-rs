mod mnist;
mod tinygrad;

use mnist::MnistTrainData;
use tinygrad::nn::neural_net;

fn main() {
    let mnist_data = MnistTrainData::load();
    let model = neural_net::Model::generate(28 * 28, 10, vec![16, 16]);
    let out = model.forward_pass(&mnist_data.get_image_nn_input(0).map(|val| f64::from(*val)));
    println!("{:?}", out);
}
