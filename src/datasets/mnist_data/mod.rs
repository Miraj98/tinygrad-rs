use crate::dataloader::{Dataloader, Dataset, Load};

#[allow(non_snake_case)]
pub fn MnistDataset() -> Dataset<[usize; 3], f32> {
    let mnist_training = Dataloader::new("./src/datasets/mnist_data/train-images-idx3-ubyte")
        .offset(16)
        .size(60_000)
        .load::<f32>((784, 1), |x| if x > 0. { x / 256. } else { x });
    let mnist_labels = Dataloader::new("./src/datasets/mnist_data/train-labels-idx1-ubyte")
        .offset(8)
        .size(60_000)
        .load_with::<f32>((10, 1), |(i, val), vec| vec[10 * i + val as usize] = 1.);

    Dataset::new((mnist_training, mnist_labels))
}
