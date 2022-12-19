use std::{
    fs::{self, File},
    io::Seek,
    ops::Range,
    os::unix::prelude::FileExt,
    path::Path,
};
use tensor_rs::{
    dim::{Dimension, Ix3},
    DataElement, Tensor,
};

pub struct Dataset<E>
where
    E: DataElement,
{
    training: Tensor<Ix3, E>,
    labels: Tensor<Ix3, E>,
}

pub struct Dataloader {
    path: &'static str,
    offset_bytes: u64,
    size: usize,
}

impl Dataloader {
    pub fn new(path: &'static str) -> Dataloader {
        Dataloader {
            path,
            size: 0,
            offset_bytes: 0,
        }
    }

    pub fn offset(mut self, x: u64) -> Self {
        self.offset_bytes = x;
        self
    }

    pub fn size(mut self, n: usize) -> Self {
        self.size = n;
        self
    }

    fn load_buf<E>(&mut self, total_bytes: usize) -> Vec<E>
    where
        E: DataElement,
    {
        let mut buf = vec![0u8; total_bytes];
        let file = File::open(self.path).unwrap();
        file.read_exact_at(&mut buf[..], self.offset_bytes).unwrap();
        buf.iter().map(|x| E::from_u8(*x)).collect::<Vec<E>>()
    }
}

pub trait Load<Rhs> {
    type Dim: Dimension;
    fn load<E>(&mut self, dim: Rhs) -> Tensor<Self::Dim, E>
    where
        E: DataElement;
}

impl Load<usize> for Dataloader {
    type Dim = [usize; 2];
    fn load<E>(&mut self, dim: usize) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(self.load_buf(self.size * dim), [self.size, dim])
    }
}

impl Load<(usize, usize)> for Dataloader {
    type Dim = [usize; 3];
    fn load<E>(&mut self, dim: (usize, usize)) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(self.load_buf(self.size * dim.0 * dim.1), [self.size, dim.0, dim.1])
    }
}

impl Load<(usize, usize, usize)> for Dataloader {
    type Dim = [usize; 4];
    fn load<E>(&mut self, dim: (usize, usize, usize)) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(self.load_buf(self.size * dim.0 * dim.1 * dim.2), [self.size, dim.0, dim.1, dim.2])
    }
}

impl Load<(usize, usize, usize, usize)> for Dataloader {
    type Dim = [usize; 5];
    fn load<E>(&mut self, dim: (usize, usize, usize, usize)) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(self.load_buf(self.size * dim.0 * dim.1 * dim.2 * dim.3), [self.size, dim.0, dim.1, dim.2, dim.3])
    }
}

impl Load<(usize, usize, usize, usize, usize)> for Dataloader {
    type Dim = [usize; 6];
    fn load<E>(&mut self, dim: (usize, usize, usize, usize, usize)) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf(self.size * dim.0 * dim.1 * dim.2 * dim.3 * dim.4),
            [self.size, dim.0, dim.1, dim.2, dim.3, dim.4],
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::dataloader::*;
    #[test]
    fn mnist() {
        let mnist_data = Dataloader::new("./src/datasets/mnist_data/train-images-idx3-ubyte")
            .offset(16)
            .size(60_000)
            .load::<f32>((784, 1));
        assert_eq!(mnist_data.dim()[0], 60_000);
        assert_eq!(mnist_data.dim()[1], 784);
        assert_eq!(mnist_data.dim()[2], 1);
    }
}
