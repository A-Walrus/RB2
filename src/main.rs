use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read, Write},
    time::Instant,
};

use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use num_complex::Complex;

fn main() -> io::Result<()> {
    const REPETITIONS: u32 = 10; // in python version called n

    const THREADS: usize = 1000;
    const ITERATIONS: usize = 100;
    const N: usize = 11520;

    let lut = get_lut();

    let start = Instant::now();
    for _ in 0..REPETITIONS {
        let randoms = Array::random((ITERATIONS, THREADS), Uniform::new(0, N)); // in python version called c
        let mut states: Vec<usize> = vec![0; THREADS]; // in python version called s

        for row in randoms.axis_iter(Axis(0)) {
            for (state, change) in states.iter_mut().zip(row) {
                *state = *(lut.get((*state, *change)).unwrap()) as usize;
            }
        }
    }

    let elapsed = Instant::now() - start;

    println!(
        "Execution time is {} ns per Clifford",
        (elapsed / REPETITIONS / THREADS as u32 / ITERATIONS as u32).as_nanos()
    );

    Ok(())
}

type Matrix = Array2<Complex<i8>>;

macro_rules! c {
    (0) => {
        Complex::new(0, 0)
    };
    (i) => {
        Complex::new(0, 1)
    };
    (-i) => {
        Complex::new(0, -1)
    };
    ($real:literal + i) => {
        Complex::new($real, 1)
    };
    ($real:literal - i) => {
        Complex::new($real, -1)
    };
    ($imag:literal i) => {
        Complex::new(0, $imag)
    };
    ($real:literal + $imag:literal i) => {
        Complex::new($real, $imag)
    };
    ($real:literal - $imag:literal i) => {
        Complex::new($real, -$imag)
    };
    ($real:literal) => {
        Complex::new($real, 0)
    };
}

const PHASES: [Complex<i8>; 4] = [c!(1), c!(i), c!(-1), c!(-i)];

fn calc_offset(m: &Matrix) -> usize {
    let mut first_non_zero = *m
        .iter()
        .find(|c| **c != c!(0))
        .expect("There should be at least one non-zero value in Matrix");
    let mut count = 0;
    while !(first_non_zero.re >= 0 && first_non_zero.im > 0) {
        first_non_zero *= c!(i);
        count += 1;
    }
    count
}

fn canonicalize(m: &mut Matrix) {
    let i = calc_offset(m);
    m.map_inplace(|c| *c *= PHASES[i])
}

fn canonicalized(m: &Matrix) -> Matrix {
    let i = calc_offset(m);
    m.map(|c| c * PHASES[i])
}

fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let mul = a.dot(b);
    debug_assert!(!mul.iter().any(|v| v.re % 2 != 0 || v.im % 2 != 0)); // There should be no odd numbers at this point
    mul / 2
}

fn push(map: &mut HashMap<Matrix, usize>, key: &Matrix) -> bool {
    let canonical = canonicalized(key);
    if !map.contains_key(&canonical) {
        let id = map.len();
        map.insert(canonical, id);
        true
    } else {
        false
    }
}

const PATH: &str = "lut";

fn save_lut(lut: &Array2<u16>) -> io::Result<()> {
    let slice = lut.as_slice().unwrap();
    let byte_slice =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 2) };

    let mut f = File::create(PATH)?;
    f.write_all(byte_slice)?;

    Ok(())
}

fn read_lut() -> io::Result<Array2<u16>> {
    let mut f = File::open(PATH)?;
    let mut vec = Vec::new();
    f.read_to_end(&mut vec)?;
    let slice = unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u16, vec.len() / 2) };
    let len = 11520;
    let vec = slice.to_vec();
    let lut = Array2::from_shape_vec((len, len), vec).unwrap();

    Ok(lut)
}

fn get_lut() -> Array2<u16> {
    match read_lut() {
        Ok(lut) => lut,
        Err(_) => {
            let lut = generate_lut();
            save_lut(&lut).unwrap();
            lut
        }
    }
}

fn generate_lut() -> Array2<u16> {
    let map = generate_map();

    let len = map.len();
    let mut keys: Vec<_> = map.keys().collect();
    keys.sort_by_key(|m| map.get(m).unwrap());

    Array2::from_shape_fn((len, len), |(i, j)| {
        let mut m = dot(keys[i], keys[j]);
        canonicalize(&mut m);
        let id = map.get(&m).unwrap();
        *id as u16
    })
}

fn generate_map() -> HashMap<Matrix, usize> {
    let originals = vec![
        sz1(),
        sx1(),
        sz2(),
        sx2(),
        cnot(),
        //
    ];

    let mut map = HashMap::new();
    for m in &originals {
        push(&mut map, m);
    }

    let mut new: Vec<_> = map.keys().cloned().collect();
    let mut newer: Vec<_> = Vec::new();
    loop {
        dbg!(map.len(), new.len(), newer.len());
        dbg!();
        for m1 in &new {
            for m2 in &originals {
                let m = dot(m1, m2);
                if push(&mut map, &m) {
                    newer.push(m);
                }
            }
        }
        if newer.is_empty() {
            break;
        }
        std::mem::swap(&mut new, &mut newer);
        newer.clear();
    }

    dbg!(map.len());
    dbg!(map.len() / 4);
    map
}

#[allow(unused)]
fn id() -> Matrix {
    Array2::from_diag_elem(4, 1.into()) * 2
}

fn sz1() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(i), c!(0), c!(0)],
        [c!(0), c!(0), c!(1), c!(0)],
        [c!(0), c!(0), c!(0), c!(i)],
    ]) * 2
}

fn sx1() -> Matrix {
    arr2(&[
        [c!(1 + i), c!(1 - i), c!(0), c!(0)],
        [c!(1 - i), c!(1 + i), c!(0), c!(0)],
        [c!(0), c!(0), c!(1 + i), c!(1 - i)],
        [c!(0), c!(0), c!(1 - i), c!(1 + i)],
    ])
}

fn sx2() -> Matrix {
    arr2(&[
        [c!(1 + i), c!(0), c!(1 - i), c!(0)],
        [c!(0), c!(1 + i), c!(0), c!(1 - i)],
        [c!(1 - i), c!(0), c!(1 + i), c!(0)],
        [c!(0), c!(1 - i), c!(0), c!(1 + i)],
    ])
}

fn sz2() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(1), c!(0), c!(0)],
        [c!(0), c!(0), c!(i), c!(0)],
        [c!(0), c!(0), c!(0), c!(i)],
    ]) * 2
}

fn cnot() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(1), c!(0), c!(0)],
        [c!(0), c!(0), c!(0), c!(1)],
        [c!(0), c!(0), c!(1), c!(0)],
    ]) * 2
}
