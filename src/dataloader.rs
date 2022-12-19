use std::{fs::File, os::unix::prelude::FileExt};
use tensor_rs::{
    dim::{Dimension, Ix3},
    DataElement, Tensor, TensorView,
};

pub struct Dataset<S, E>
where
    E: DataElement,
    S: Dimension
{
    training_set: Tensor<S, E>,
    labels: Tensor<S, E>,
    test_set: Option<Tensor<S, E>>,
}

impl<E, S> Dataset<S, E>
where
    E: DataElement,
    S: Dimension
{
    pub fn new(args: (Tensor<S, E>, Tensor<S, E>, Option<Tensor<S, E>>)) -> Self {
        Dataset { training_set: args.0, labels: args.1, test_set: args.2 }
    }
}

pub struct DatasetIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    dataset: &'a Dataset<S, E>,
    index: usize
}

impl<'a, S, E> Iterator for DatasetIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    type Item = (TensorView<'a, S::Smaller, E>, TensorView<'a, S::Smaller, E>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.training_set.dim()[0] { return None; }
        let val = (self.dataset.training_set.outer_dim(self.index), self.dataset.labels.outer_dim(self.index));
        self.index += 1;
        return Some(val);
    }
}

impl<'a, S, E> IntoIterator for &'a Dataset<S, E>
where
    S: Dimension,
    E: DataElement
{
    type IntoIter = DatasetIterator<'a, S, E>;
    type Item = (TensorView<'a, S::Smaller, E>, TensorView<'a, S::Smaller, E>);
    fn into_iter(self) -> Self::IntoIter {
        DatasetIterator { dataset: self, index: 0 }
    }
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

    fn load_buf<E>(&mut self, total_bytes: usize, normaliser: impl Fn(E) -> E) -> Vec<E>
    where
        E: DataElement,
    {
        let mut buf = vec![0u8; total_bytes];
        let file = File::open(self.path).unwrap();
        file.read_exact_at(&mut buf[..], self.offset_bytes).unwrap();
        buf.iter()
            .map(|x| normaliser(E::from_u8(*x)))
            .collect::<Vec<E>>()
    }

    fn load_buf_with<E>(
        &mut self,
        in_size: usize,
        out_size: usize,
        mut f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Vec<E>
    where
        E: DataElement,
    {
        let zero = E::zero();
        let mut vec = vec![zero; out_size];
        let mut buf = vec![0u8; in_size];
        let file = File::open(self.path).unwrap();
        file.read_exact_at(&mut buf[..], self.offset_bytes).unwrap();
        buf.iter()
            .enumerate()
            .for_each(|(i, val)| f((i, *val), &mut vec));
        vec
    }
}

pub trait Load<Rhs> {
    type Dim: Dimension;
    fn load<E>(&mut self, dim: Rhs, normaliser: impl Fn(E) -> E) -> Tensor<Self::Dim, E>
    where
        E: DataElement;
    fn load_with<E>(
        &mut self,
        dim: Rhs,
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement;
}

impl Load<usize> for Dataloader {
    type Dim = [usize; 2];
    fn load<E>(&mut self, dim: usize, f: impl Fn(E) -> E) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(self.load_buf(self.size * dim, f), [self.size, dim])
    }

    fn load_with<E>(
        &mut self,
        dim: usize,
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf_with(in_size, self.size * dim, f),
            [self.size, dim],
        )
    }
}

impl Load<(usize, usize)> for Dataloader {
    type Dim = [usize; 3];
    fn load<E>(&mut self, dim: (usize, usize), f: impl Fn(E) -> E) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf(self.size * dim.0 * dim.1, f),
            [self.size, dim.0, dim.1],
        )
    }

    fn load_with<E>(
        &mut self,
        dim: (usize, usize),
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf_with(in_size, self.size * dim.0 * dim.1, f),
            [self.size, dim.0, dim.1],
        )
    }
}

impl Load<(usize, usize, usize)> for Dataloader {
    type Dim = [usize; 4];
    fn load<E>(&mut self, dim: (usize, usize, usize), f: impl Fn(E) -> E) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf(self.size * dim.0 * dim.1 * dim.2, f),
            [self.size, dim.0, dim.1, dim.2],
        )
    }

    fn load_with<E>(
        &mut self,
        dim: (usize, usize, usize),
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf_with(in_size, self.size * dim.0 * dim.1 * dim.2, f),
            [self.size, dim.0, dim.1, dim.2],
        )
    }
}

impl Load<(usize, usize, usize, usize)> for Dataloader {
    type Dim = [usize; 5];
    fn load<E>(
        &mut self,
        dim: (usize, usize, usize, usize),
        f: impl Fn(E) -> E,
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf(self.size * dim.0 * dim.1 * dim.2 * dim.3, f),
            [self.size, dim.0, dim.1, dim.2, dim.3],
        )
    }

    fn load_with<E>(
        &mut self,
        dim: (usize, usize, usize, usize),
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf_with(in_size, self.size * dim.0 * dim.1 * dim.2 * dim.3, f),
            [self.size, dim.0, dim.1, dim.2, dim.3],
        )
    }
}

impl Load<(usize, usize, usize, usize, usize)> for Dataloader {
    type Dim = [usize; 6];
    fn load<E>(
        &mut self,
        dim: (usize, usize, usize, usize, usize),
        f: impl Fn(E) -> E,
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf(self.size * dim.0 * dim.1 * dim.2 * dim.3 * dim.4, f),
            [self.size, dim.0, dim.1, dim.2, dim.3, dim.4],
        )
    }

    fn load_with<E>(
        &mut self,
        dim:(usize, usize, usize, usize, usize),
        in_size: usize,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement,
    {
        Tensor::from_vec(
            self.load_buf_with(in_size, self.size * dim.0 * dim.1 * dim.2 * dim.3 * dim.4, f),
            [self.size, dim.0, dim.1, dim.2, dim.3, dim.4],
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::dataloader::*;
    #[test]
    fn mnist() {
        let mnist_training = Dataloader::new("./src/datasets/mnist_data/train-images-idx3-ubyte")
            .offset(16)
            .size(60_000)
            .load::<f32>((784, 1), |x| if x > 0. { x / 256. } else { x });
        let mnist_labels = Dataloader::new("./src/datasets/mnist_data/train-labels-idx1-ubyte")
            .offset(8)
            .size(60_000)
            .load_with::<f32>((10, 1), 60_000, |(i, val), vec| { vec[10 * i + val as usize] = 1. });

        assert_eq!(mnist_training.dim()[0], 60_000);
        assert_eq!(mnist_training.dim()[1], 784);
        assert_eq!(mnist_training.dim()[2], 1);

        assert_eq!(mnist_labels.dim()[0], 60_000);
        assert_eq!(mnist_labels.dim()[1], 10);
        assert_eq!(mnist_labels.dim()[2], 1);
    }
}
