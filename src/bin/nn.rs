use tinygrad_rust::{
    datasets::{mnist, PX_SIZE},
    neural_net, nn,
};

fn main() {
    let mnist_dataloader = mnist::load_data();
    let batch_size = 10;
    let epochs = 30;
    let mut model = neural_net::Model::new(
        PX_SIZE * PX_SIZE,
        10,
        vec![16, 16],
        mnist_dataloader,
        nn::CrossEntropyLoss("mean"),
    );

    model.train(batch_size, epochs)
}
