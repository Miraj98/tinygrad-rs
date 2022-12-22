pub mod linear;

pub use linear::Linear;
use tensor_rs::{
    TensorBase,
    impl_constructors::TensorConstructors,
    impl_processing_ops::{Conv2d as Convolution2d}, DataBuffer, DataElement, Tensor, dim::{Ix4, Ix3},
};

pub trait Layer<Input> {
    type Output;

    fn forward(&self, input: Input) -> Self::Output;
}


pub struct Conv2d<E> 
where
    E: DataElement
{
    w: Tensor<Ix4, E>,
    b: Tensor<Ix4, E>,
    strides: (usize, usize),
}

impl<E> Conv2d<E>
where
    E: DataElement + 'static
{
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            w: Tensor::randn([output_channels, input_channels, kernel_size.0, kernel_size.1]).requires_grad(true),
            b: Tensor::randn([output_channels, 1, 1, 1]).requires_grad(true),
            strides,
        }
    }

    pub fn bias(&self) -> &Tensor<Ix4, E> {
        &self.b
    }

    pub fn weight(&self) -> &Tensor<Ix4, E> {
        &self.w
    }

    pub fn strides(&self) -> (usize, usize) {
        self.strides
    }
}

impl<B, E> Layer<&TensorBase<Ix3, B>> for Conv2d<E>
where
    B: DataBuffer<Item = E> + 'static,
    E: DataElement + 'static
{
    type Output = Tensor<Ix3, E>;
    fn forward(&self, input: &TensorBase<Ix3, B>) -> Self::Output {
       input.conv2d(&self.w, (1, 1)) 
    }
}



