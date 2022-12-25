use super::Layer;
use tensor_rs::{
    dim::{Ix, Ix2},
    impl_binary_ops::TensorBinaryOps,
    impl_constructors::TensorConstructors,
    impl_processing_ops::Matmul,
    DataElement, Tensor, TensorView,
};

pub struct Linear<A>
where
    A: DataElement,
{
    w: Tensor<Ix2, A>,
    b: Tensor<Ix2, A>,
}

impl<A> Linear<A>
where
    A: DataElement,
{
    pub fn new(d1: Ix, d2: Ix) -> Self {
        Self {
            w: Tensor::randn([d1, d2]).requires_grad(true),
            b: Tensor::randn([d1, 1]).requires_grad(true),
        }
    }

    pub fn bias(&self) -> &Tensor<Ix2, A> {
        &self.b
    }

    pub fn weight(&self) -> &Tensor<Ix2, A> {
        &self.w
    }

    pub fn bias_mut(&mut self) -> &mut Tensor<Ix2, A> {
        &mut self.b
    }

    pub fn weight_mut(&mut self) -> &mut Tensor<Ix2, A> {
        &mut self.w
    }

    pub fn parameters(&self) -> LinearLayerIterator<'_, A> {
        LinearLayerIterator::new(self)
    }

    pub fn zeros(&self) -> (Tensor<Ix2, A>, Tensor<Ix2, A>) {
        (Tensor::zeros(self.w.dim()), Tensor::zeros(self.b.dim()))
    }
}

pub struct LinearLayerIterator<'a, A>
where
    A: DataElement,
{
    index: usize,
    layer: &'a Linear<A>,
}

impl<'a, A> LinearLayerIterator<'a, A>
where
    A: DataElement,
{
    pub fn new(layer: &'a Linear<A>) -> Self {
        Self { index: 0, layer }
    }
}

impl<'a, A> Iterator for LinearLayerIterator<'a, A>
where
    A: DataElement,
{
    type Item = &'a Tensor<Ix2, A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.index >= 2 {
            return None;
        }
        match self.index {
            0 => Some(&self.layer.w),
            1 => Some(&self.layer.b),
            _ => unreachable!(),
        }
    }
}

impl Layer<&Tensor<Ix2>> for Linear<f32> {
    type Output = Tensor<Ix2>;

    fn forward(&self, input: &Tensor<Ix2>) -> Self::Output {
        self.w.matmul(input).add(&self.b)
    }

    fn with_training(mut self, train: bool) -> Self {
        self.w = self.w.requires_grad(train);
        self.b = self.b.requires_grad(train);
        self
    }
}

impl Layer<Tensor<Ix2>> for Linear<f32> {
    type Output = Tensor<Ix2>;

    fn forward(&self, input: Tensor<Ix2>) -> Self::Output {
        self.w.matmul(input).add(&self.b)
    }

    fn with_training(mut self, train: bool) -> Self {
        self.w = self.w.requires_grad(train);
        self.b = self.b.requires_grad(train);
        self
    }
}

impl Layer<TensorView<Ix2, f32>> for Linear<f32> {
    type Output = Tensor<Ix2>;

    fn forward(&self, input: TensorView<Ix2, f32>) -> Self::Output {
        self.w.matmul(input).add(&self.b)
    }

    fn with_training(mut self, train: bool) -> Self {
        self.w = self.w.requires_grad(train);
        self.b = self.b.requires_grad(train);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, Linear};
    use tensor_rs::{impl_reduce_ops::ReduceOps, Tensor};

    #[test]
    fn linear() {
        let m = Linear::new(30, 2);
        let a = Tensor::from_vec(vec![1., 2.], [2, 1]);
        let o = m.forward(&a);
        let loss = o.sum();
        let g = loss.backward();
        let w_grad = g.grad(m.weight());
        let b_grad = g.grad(m.bias());
        println!("{:?}", w_grad);
        println!("{:?}", b_grad);
    }
}
