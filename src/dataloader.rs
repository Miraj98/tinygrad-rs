use std::{fs::File, os::unix::prelude::FileExt, ops::Range};
use tensor_rs::{
    dim::{Dimension, IntoDimension},
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

    pub fn iter(&self) -> DatasetIterator<'_, S, E> {
        DatasetIterator::new(self)
    }

    pub fn batch_iter(&self, batch_size: usize) -> DatasetBatchIterator<'_, S, E> {
        let nbatches = self.training_set.dim()[0] / batch_size;
        DatasetBatchIterator { dataset: self, nbatches, batch_size, batch_idx: 0 }
    }
}

pub struct DatasetBatchIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    dataset: &'a Dataset<S, E>,
    nbatches: usize,
    batch_size: usize,
    batch_idx: usize,
}

impl<'a, S, E> Iterator for DatasetBatchIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    type Item = DatasetIterator<'a, S, E>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_idx >= self.nbatches - 1 { return None }
        let val = DatasetIterator::new(self.dataset).range((self.batch_size * self.batch_idx)..(self.batch_size * self.batch_idx + self.batch_size));
        self.batch_idx += 1;
        return Some(val);
    }

    fn count(self) -> usize
        where
            Self: Sized, {
        return self.nbatches;
    }
}

pub struct DatasetIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    dataset: &'a Dataset<S, E>,
    index: usize,
    range: Range<usize>,
}

impl<'a, S, E> DatasetIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    pub fn new(dataset: &'a Dataset<S, E>) -> Self {
        Self { dataset, index: 0, range: Range { start: 0, end: dataset.training_set.dim()[0] } }
    }

    pub fn range(mut self, range: Range<usize>) -> Self {
        assert!(range.start < range.end);
        assert!(range.end < self.dataset.training_set.dim()[0]);
        self.range = range;
        self
    }
}

impl<'a, S, E> Iterator for DatasetIterator<'a, S, E>
where
    S: Dimension,
    E: DataElement
{
    type Item = (TensorView<S::Smaller, E>, TensorView<S::Smaller, E>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.range.end - 1 { return None; }
        let val = (self.dataset.training_set.outer_dim(self.index), self.dataset.labels.outer_dim(self.index));
        self.index += 1;
        return Some(val);
    }

    fn count(self) -> usize
        where
            Self: Sized, {
        self.dataset.training_set.dim()[0]
    }
}

impl<'a, S, E> IntoIterator for &'a Dataset<S, E>
where
    S: Dimension,
    E: DataElement
{
    type IntoIter = DatasetIterator<'a, S, E>;
    type Item = (TensorView<S::Smaller, E>, TensorView<S::Smaller, E>);
    fn into_iter(self) -> Self::IntoIter {
        DatasetIterator::new(self)
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
        total_bytes: usize,
        mut f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Vec<E>
    where
        E: DataElement,
    {
        let zero = E::zero();
        let mut vec = vec![zero; total_bytes];
        let mut buf = vec![0u8; self.size];
        let file = File::open(self.path).unwrap();
        file.read_exact_at(&mut buf[..], self.offset_bytes).unwrap();
        buf.iter()
            .enumerate()
            .for_each(|(i, val)| f((i, *val), &mut vec));
        vec
    }
}

pub trait Load<Rhs: IntoDimension> {
    type Dim: Dimension;
    fn load<E>(&mut self, dim: Rhs, normaliser: impl Fn(E) -> E) -> Tensor<Self::Dim, E>
    where
        E: DataElement;
    fn load_with<E>(
        &mut self,
        dim: Rhs,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E>
    where
        E: DataElement;
}

impl<Rhs: IntoDimension> Load<Rhs> for Dataloader {
    type Dim = <Rhs::Dim as Dimension>::Larger;

    fn load<E>(
        &mut self,
        dim: Rhs,
        normaliser: impl Fn(E) -> E
    ) -> Tensor<Self::Dim, E> where E: DataElement {
       let d = dim.into_dimension().expand(self.size);
       let total_bytes = d.count();
       Tensor::from_vec(self.load_buf(total_bytes, normaliser), d)
    }

    fn load_with<E>(
        &mut self,
        dim: Rhs,
        f: impl FnMut((usize, u8), &mut Vec<E>),
    ) -> Tensor<Self::Dim, E> where E: DataElement {
       let d = dim.into_dimension().expand(self.size);
       let total_bytes = d.count();
       Tensor::from_vec(self.load_buf_with(total_bytes, f), d)
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
            .load_with::<f32>((10, 1), |(i, val), vec| { vec[10 * i + val as usize] = 1. });

        assert_eq!(mnist_training.dim()[0], 60_000);
        assert_eq!(mnist_training.dim()[1], 784);
        assert_eq!(mnist_training.dim()[2], 1);

        assert_eq!(mnist_labels.dim()[0], 60_000);
        assert_eq!(mnist_labels.dim()[1], 10);
        assert_eq!(mnist_labels.dim()[2], 1);
    }
}
