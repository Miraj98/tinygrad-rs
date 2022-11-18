use ndarray::{array, Array2, s, Axis};

fn main() {
    let a = array![[1,2, 3], [2,3, 4], [4,5,6]];

    let mut filter = array![[1,2], [3,4]];
    let mut filter_view = filter.view_mut();
    filter_view.invert_axis(Axis(0));
    filter_view.invert_axis(Axis(1));
    println!("{:?}", filter_view);
    println!("{:?}", filter);
    // let px = filter.shape()[0] - 1;
    // let py = filter.shape()[1] - 1;
    // println!("px, py {}, {}", px, py);
    // let mut padded = Array2::<i32>::zeros((2 * px + a.dim().0, 2 * py + a.dim().1 ));
    // let mut padded_view = padded.view_mut();
    // let mut s = padded_view.slice_mut(s![py..py + a.dim().0, px..px + a.dim().1]);
    // s.assign(&a);

    // println!("{:?}", padded);
}