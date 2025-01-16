use ndarray::{s, Array, ArrayView1, ArrayView2, ArrayView3, Array1, Array2, Array3, Axis};
use nalgebra::{DMatrix, Scalar};
use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use num::{Zero, One, ToPrimitive};


pub fn weighted_random_choice(
    weights: &ArrayView1<u64>,
    rng: &mut impl Rng,
) -> usize {
    let dist = WeightedIndex::new(weights).unwrap();
    dist.sample(rng)
}

fn swap_rows<T>(array: &mut Array2<T>, row1: usize, row2: usize)
where
    T: Clone,
{
    for mut col in array.columns_mut() {
        col.swap(row1, row2);
    }
}

fn swap_columns<T>(array: &mut Array2<T>, col1: usize, col2: usize) 
where
    T: Clone,
{
    for mut row in array.rows_mut() {
        row.swap(col1, col2);
    }
}

fn ndarray_to_dmatrix<T>(ndarray: &Array2<T>) -> DMatrix<T> 
where
    T: Scalar + Clone,
{
    let ndarray_c = Array::from_shape_vec(ndarray.raw_dim(), ndarray.iter().cloned().collect()).unwrap();
    let data: Vec<T> = ndarray_c.iter().cloned().collect();
    
    let rows = ndarray_c.nrows();
    let cols = ndarray_c.ncols();
    DMatrix::from_row_slice(rows, cols, &data)
}

fn dmatrix_to_ndarray<T>(dmatrix: &DMatrix<T>) -> Array2<T> 
where
    T: Scalar + Clone,
{
    let data: Vec<T> = dmatrix.iter().cloned().collect();
    
    Array2::from_shape_vec((dmatrix.nrows(), dmatrix.ncols()), data).unwrap().t().to_owned()
}

fn rand_arborescence(
    adj_in: &ArrayView2<usize>, 
    start_idx: usize, 
    rng: &mut impl Rng,
) -> Array2<usize> {

    let mut adj = adj_in.mapv(|x| x as f64);
    
    let n = adj.nrows();

    // Swap rows and columns to bring start_idx to the top left
    swap_rows(&mut adj, 0, start_idx);
    swap_columns(&mut adj, 0, start_idx);

    // Set diagonal elements to zero
    for i in 0..n {
        adj[(i, i)] = 0.0;
    }

    // Compute the Laplacian matrix
    let mut lap = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let col_sum: f64 = adj.column(i).sum();
        lap[(i, i)] = col_sum;
    }
    lap -= &adj;

    // Extract minor of Laplacian and compute determinant
    let lap_minor = lap.slice(s![1.., 1..]).to_owned();
    let mut det = ndarray_to_dmatrix(&lap_minor).determinant();

    // Compute the inverse of the minor
    let mut ki = dmatrix_to_ndarray(&ndarray_to_dmatrix(&lap_minor).try_inverse().unwrap());

    let mut out = Array2::<usize>::zeros((n, n));

    for c in 1..n {
        let vt_ki = ki.slice(s![c - 1..c, ..]).to_owned();
        let mut u1 = -lap.slice(s![.., c..c + 1]).to_owned();
        u1[(c, 0)] += 1.;

        let vt_ki_u1 = vt_ki.dot(&u1.slice(s![1.., ..]));

        let mut updates = Array2::<f64>::zeros((n, 1));
        updates.slice_mut(s![1.., ..]).assign(&(-vt_ki.t().to_owned()));

        let vt_ki_u_p1s = 1. + vt_ki_u1 + updates;
        let new_dets = det * &vt_ki_u_p1s;
        let counts = adj.slice(s![..,c..c + 1]).to_owned() * &new_dets;

        let counts_u64: Array1<u64> = counts.slice(s![..,0]).mapv(|x| x.round() as u64);

        let edge = weighted_random_choice(&counts_u64.view(), rng);

        out[(edge, c)] = 1;
        det = new_dets[(edge, 0)];

        let mut u = u1.clone();
        u.slice_mut(s![edge, ..]).map_inplace(|x| *x -= 1.);

        let ki_u = ki.dot(&u.slice(s![1.., ..]));
        let ki_u_vt_ki = ki_u.dot(&vt_ki);
        let ki_update = ki_u_vt_ki / vt_ki_u_p1s[(edge, 0)];
        ki = ki - ki_update;

        adj.slice_mut(s![.., c]).fill(0.);
        adj[(edge, c)] = 1.;
        lap.slice_mut(s![.., c]).fill(0.);
        lap[(edge, c)] = -1.;
        lap[(c, c)] = 1.;
    }

    swap_rows(&mut out, 0, start_idx);
    swap_columns(&mut out, 0, start_idx);

    out
}


pub fn eulerian_walk(
    adj: &ArrayView2<usize>,
    tree: &ArrayView2<usize>,
    start_idx: usize,
    walk_len: usize,
    rng: &mut impl Rng,
    alphabet_size: usize,
) -> Array1<usize> {

    let mut non_tree = adj - tree;
    let mut non_tree_sum: Array1<usize> = non_tree.sum_axis(Axis(1));

    // Precompute tree indices
    let tree_inds: Array1<usize> = tree
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &value)| j as usize * value)
                .sum::<usize>() as usize
        })
        .collect();

    // Precompute WeightedIndex for each row of non_tree
    let mut distributions: Vec<Option<WeightedIndex<usize>>> = (0..alphabet_size)
        .map(|i| {
            if non_tree_sum[i] > 0 {
                Some(WeightedIndex::new(&non_tree.row(i)).unwrap())
            } else {
                None
            }
        })
        .collect();

    // Initialize walk array
    let mut walk = Array1::zeros(walk_len);
    walk[0] = start_idx;

    let mut idx = start_idx;

    for i in 1..walk_len {
        let idx_prev = idx;

        if non_tree_sum[idx] == 0 {
            idx = tree_inds[idx];
        } else {
            // Sample next index
            let dist = distributions[idx].as_mut().unwrap();
            idx = dist.sample(rng);

            // Update non_tree matrix and WeightedIndex
            non_tree_sum[idx_prev] -= 1;
            non_tree[[idx_prev, idx]] -= 1;

            if non_tree_sum[idx_prev] > 0 {
                let _ = distributions[idx_prev]
                    .as_mut()
                    .unwrap()
                    .update_weights(&[(idx, &non_tree[[idx_prev, idx]])]);
            } else {
                distributions[idx_prev] = None;
            }
        }

        walk[i] = idx;
    }

    walk

}


fn argmax<T>(array: &ArrayView1<T>) -> usize
where
    T: PartialOrd + Clone,
{
    let mut max_index = 0;
    let mut max_value = array[max_index].clone();

    for (i, value) in array.iter().enumerate() {
        if value > &max_value {
            max_value = value.clone();
            max_index = i;
        }
    }

    max_index
}


fn inds_to_onehot<T>(
    inds: &ArrayView1<usize>, 
    out_map: &ArrayView1<usize>,
    alphabet_size: usize
) -> Array2<T> 
where T: Clone + ToPrimitive + Zero + One
{
    let n = inds.len();
    let mut onehot = Array2::<T>::zeros((n, alphabet_size));

    for (i, &ind) in inds.iter().enumerate() {
        let ind_out = out_map[ind];
        onehot[(i, ind_out)] = T::one();
    }

    onehot
}

fn shuffle<T>(
    seq_in: &ArrayView2<T>, 
    rng: &mut impl Rng
) -> Array2<T> 
where T: Clone + ToPrimitive + Zero + One
{
    let seq = seq_in.mapv(|x| x.to_usize().unwrap());

    let freqs = seq.sum_axis(Axis(0));
    let mask = freqs.mapv(|x| x > 0);
    let mask_sum = mask.mapv(|x| x as usize).sum();
    if mask_sum == 1 {
        return seq_in.to_owned();
    }
    let mask_inds: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| if value { Some(index) } else { None })
        .collect();

    let alphabet_size = seq.shape()[1];

    let seq = seq.select(Axis(1), &mask_inds);

    let (n, z) = (seq.shape()[0], seq.shape()[1]);

    let adj = seq.slice(s![..-1, ..]).t().dot(&seq.slice(s![1.., ..]));

    let start_idx = argmax(&seq.row(0));
    let end_idx = argmax(&seq.row(n - 1));

    let tree_t = rand_arborescence(&adj.t().view(), end_idx, rng);
    let tree = tree_t.t().to_owned();

    let walk_inds = eulerian_walk(&adj.view(), &tree.view(), start_idx, n, rng, z);

    let out_map = Array1::from(mask_inds);
    let walk: Array2<T> = inds_to_onehot(&walk_inds.view(), &out_map.view(), alphabet_size);

    walk

}


pub fn batched_shuffle<T>( 
    seq_in: &ArrayView3<T>, 
    rng: &mut impl Rng
) -> Array3<T> 
where T: Clone + ToPrimitive + Zero + One
{
    let num_seqs = seq_in.shape()[0];
    let output_len = seq_in.shape()[1];
    let alphabet_size = seq_in.shape()[2];

    let mut output = Array3::<T>::zeros((num_seqs, output_len, alphabet_size));
    for seq_idx in 0..num_seqs {
        let seq_slice = seq_in.index_axis(Axis(0), seq_idx);
        let mut output_row = output.index_axis_mut(Axis(0), seq_idx);
        let walk_result = shuffle(&seq_slice, rng);
        output_row.assign(&walk_result);
    }

    output
}