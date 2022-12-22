use tensor_rs::{
    dim::Dimension, impl_binary_ops::TensorBinaryOps, impl_reduce_ops::ReduceOps,
    impl_unary_ops::TensorUnaryOps, Tensor, impl_constructors::TensorConstructors, TensorView,
};

#[allow(non_snake_case)]
#[inline]
pub fn MSELoss<S>() -> impl Fn(Tensor<S>, TensorView<S>) -> Tensor<[usize; 0]>
where
    S: Dimension + 'static,
{
    |output, target| {
        let l = output.sub(&target).square();
        let loss = l.mean();
        loss
    }
}

#[allow(non_snake_case)]
#[inline]
pub fn CrossEntropyLoss<S>() -> impl Fn(Tensor<S>, TensorView<S>) -> Tensor<[usize; 0]>
where
    S: Dimension + 'static,
{
    |output, target| {
        let output_ln = output.ln();
        let lhs = target.mul(&output_ln);
        let ones = Tensor::ones(target.dim());
        let _a = ones.sub(&output);
        let _y = ones.sub(&target);
        let rhs = _y.mul(&_a.ln());

        let l = lhs.add(&rhs) * -1.;
        let loss = l.mean();
        loss
    }
}
