use std::rc::Rc;

use ndarray::{array, s, Array2};
use tinygrad_rust::{
    datasets::{mnist, Dataloader, PX_SIZE},
    tensor::Tensor,
};

fn main() {
    let m = mnist::load_data();
    let (mut x, y) = m.get_by_idx(0);
    x = x.into_shape((PX_SIZE, PX_SIZE)).unwrap();
    println!("{:?}", x);
    let num_kernels = 20;
    let strides = 1; 
    let k_size = (5, 5);
    let kernels: Vec<Rc<Tensor>> = (0..20).map(|_| Tensor::randn(k_size, Some(true))).collect();
    let mut conv_out: Vec<Rc<Tensor>> = (0..20)
        .map(|_| Tensor::zeros((PX_SIZE - k_size.0 + 1, PX_SIZE - k_size.0 + 1), Some(true)))
        .collect();
    let max_pool_out:Vec<Rc<Tensor>> = (0..20).map(|_| Tensor::zeros(
        ((PX_SIZE - k_size.0 + 1) / 2, (PX_SIZE - k_size.0 + 1) / 2),
        Some(true),
    )).collect();

    // for k in (0..kernels.len()) {
    //     for col in 0..(x.dim().1 - k_size.1 + 1) {
    //         for row in 0..(x.dim().0 - k_size.0 + 1) {
    //             conv_out[k] = &x.slice(s![col*k_size.0..col*k_size.0 + k_size.0, row*k_size.0..row*k_size.0 + k_size.0]).
    //         }
    //     }
    // }

    let mut x = array![
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29]
        ],
        [
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44]
        ]
    ];

    let mut x_slice = x.slice(s![2, .., ..]);
    let a = &x_slice * 2;
    println!("{:?}", a);
}
