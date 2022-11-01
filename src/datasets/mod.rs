pub mod mnist {
    use ndarray::Array2;
    use std::fs;
    use ::image::GrayImage;

    pub struct MnistData {
        raw_data: Vec<u8>,
        pub raw_labels_data: Vec<u8>,
        pub DATA_SET_SIZE: u16,
    }

    pub fn load_data() -> MnistData {
        MnistData {
            raw_data: fs::read("./src/datasets/mnist_data/train-images-idx3-ubyte").unwrap(),
            raw_labels_data: fs::read("./src/datasets/mnist_data/train-labels-idx1-ubyte").unwrap(),
            DATA_SET_SIZE: 60_000,
        }
    }

    impl MnistData {
        pub fn get_img_buffer(&self, idx: usize) -> &[u8] {
            &self.raw_data[(28 * 28 * idx + 16)..(16 + idx * 28 * 28 + 28 * 28)]
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
            Array2::from_shape_vec((28 * 28, 1), buf)
                .unwrap()
                .mapv(|val| (val as f64 / 256.))
        }

        pub fn save_as_png(&self, idx: usize) {
            let buf = self.get_img_buffer(idx).to_vec();
            let img = GrayImage::from_vec(28, 28, buf).unwrap();
            img.save_with_format(format!("{}.png", idx), image::ImageFormat::Png)
                .unwrap();
        }
    }
}
