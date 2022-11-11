use ndarray::Array2;

pub trait Dataloader {
    fn get_by_idx(&self, idx: usize) -> (Array2<f64>, Array2<f64>);
    fn get_batch(&self, batch_size: usize, batch_idx: usize) -> Vec<(Array2<f64>, Array2<f64>)>;
    fn size(&self) -> u16;
}

pub const PX_SIZE: usize = 28;

pub mod mnist {
    use ndarray::Array2;
    use std::fs;
    use ::image::GrayImage;

    use super::{Dataloader, PX_SIZE};

    pub struct MnistData {
        raw_data: Vec<u8>,
        pub raw_labels_data: Vec<u8>,
        pub dataset_size: u16,
    }

    pub fn load_data() -> MnistData {
        MnistData {
            raw_data: fs::read("./src/datasets/mnist_data/train-images-idx3-ubyte").unwrap(),
            raw_labels_data: fs::read("./src/datasets/mnist_data/train-labels-idx1-ubyte").unwrap(),
            dataset_size: 60_000,
        }
    }

    impl Dataloader for MnistData {
        fn get_by_idx(&self, idx: usize) -> (Array2<f64>, Array2<f64>) {
            return (self.get_image_nn_input(idx), self.get_image_label_vector(idx))
        }
        
        fn get_batch(&self, batch_size: usize, batch_idx: usize) -> Vec<(Array2<f64>, Array2<f64>)> {
            let mut b = Vec::<(Array2<f64>, Array2<f64>)>::new();

            if self.size() % batch_size as u16 != 0 {
                panic!("Batch size must be a whole factor of the total dataset size")
            }

            for i in 0..batch_size {
                let idx = batch_idx * batch_size + i;
                b.push(self.get_by_idx(idx))
            }

            b
        }
        
        fn size(&self) -> u16 {
            self.dataset_size
        }
    }

    impl MnistData {
        pub fn get_img_buffer(&self, idx: usize) -> &[u8] {
            &self.raw_data[(PX_SIZE * PX_SIZE * idx + 16)..(16 + idx * PX_SIZE * PX_SIZE + PX_SIZE * PX_SIZE)]
        }

        pub fn get_image_label(&self, idx: usize) -> u8 {
            self.raw_labels_data[idx + 8]
        }

        pub fn get_image_label_vector(&self, idx: usize) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((10, 1));
            out[(self.get_image_label(idx) as usize, 0)] = 1.;
            return out;
        }

        pub fn get_image_nn_input(&self, idx: usize) -> Array2<f64> {
            let buf = self.get_img_buffer(idx).to_vec();
            Array2::from_shape_vec((PX_SIZE * PX_SIZE, 1), buf)
                .unwrap()
                .mapv(|val| (val as f64 / 256.))
        }

        pub fn save_as_png(&self, idx: usize) {
            let buf = self.get_img_buffer(idx).to_vec();
            let img = GrayImage::from_vec(PX_SIZE as u32, PX_SIZE as u32, buf).unwrap();
            img.save_with_format(format!("{}.png", idx), image::ImageFormat::Png)
                .unwrap();
        }
    }
}
