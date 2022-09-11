use std::{fs};
use ndarray::Array2;

pub struct MnistTrainData {
    raw_data: Vec<u8>,
    raw_labels_data: Vec<u8>,
}

impl MnistTrainData {
    pub fn load() -> MnistTrainData {
        MnistTrainData {
            raw_data: fs::read("./mnist_data/train-images-idx3-ubyte").unwrap(),
            raw_labels_data: fs::read("./mnist_data/train-labels-idx1-ubyte").unwrap(),
        }
    }

    pub fn get_img_buffer(&self, idx: usize) -> &[u8]  {
        &self.raw_data[(idx + 16)..(idx + 28*28)]
    }

    pub fn get_image_label(&self, idx: usize) -> u8  {
        self.raw_labels_data[idx + 8]
    }

    pub fn get_image_nn_input(&self, idx: usize) -> Array2<u8>  {
        let buf = self.raw_data[(16 + idx)..(16 + idx + 28*28)].to_vec();
        Array2::from_shape_vec((28*28, 1), buf).unwrap()
    }
}
