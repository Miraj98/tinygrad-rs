use tinygrad_rust::{datasets::{mnist, PX_SIZE}, neural_net};

fn main() {
    let mnist_dataloader = mnist::load_data();
    let batch_size = 60;
    let epochs = 5;
    // let mut model = neural_net::Model::init(PX_SIZE * PX_SIZE, 10, vec![8], mnist_dataloader);

    // model.train(batch_size, epochs)
}
