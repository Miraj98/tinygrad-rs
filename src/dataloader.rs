use tensor_rs::{Tensor, DataElement, dim::Ix3};
use std::fs;

pub struct Dataset<E>
where
    E: DataElement
{
    input: Tensor<Ix3, E>,
    target: Tensor<Ix3, E>,
    batch_size: usize,
}

impl<E> Dataset<E>
where
    E: DataElement
{
    pub fn from_path(data: &str, labels: &str, f: impl Fn(Vec<u8>, Vec<u8>) -> Self) -> Self {
        let raw_data = fs::read(data).expect("The data exists");
        let raw_labels = fs::read(labels).expect("The labels exist");
        f(raw_data, raw_labels)
    }

    pub fn from_params(input: Tensor<Ix3, E>, target: Tensor<Ix3, E>, batch_size: usize) -> Self {
        Self { input, target, batch_size }
    }
}

#[cfg(test)]
mod tests {
    use tensor_rs::IntoFloat32;

    use crate::dataloader::*;
    #[test]
    fn mnist() {
        let _: Dataset<f32> = Dataset::from_path(
            "./src/datasets/mnist_data/train-images-idx3-ubyte",
            "./src/datasets/mnist_data/train-labels-idx1-ubyte",
            |mut input, target| {
                input.drain(0..16);
                let i = input.iter().map(|x| x.f32()).collect::<Vec<f32>>();
                let mut t: Vec<f32> = vec![0.; 60000*10];

                let input_tensor = Tensor::from_vec(i, [60000, 784, 1]);
                for _i in 0..60000 {
                    t[_i * 10 + target[_i + 8] as usize] = 1.;
                }
                let target_tensor = Tensor::from_vec(t, [60000, 10, 1]);
                Dataset::from_params(input_tensor, target_tensor, 10)
            }
        );
    }
}
