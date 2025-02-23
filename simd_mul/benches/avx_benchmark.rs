use ark_ff::One;
use ark_ff::UniformRand;
use mina_hasher::Fp;
use simd_mul::avx::mul_assign_fp4;
use simd_mul::avx::mul_fp4;
use simd_mul::avx::pack4fields;
use std::arch::x86_64::*;
use criterion::black_box;
use criterion::{Criterion, criterion_group, criterion_main};


fn generate_random_values() -> (Vec<Fp>, Vec<Fp>) {
    let mut rng = rand::rngs::OsRng;
    let size = 200000;
    let mut random_values1 = Vec::with_capacity(size);
    let mut random_values2 = Vec::with_capacity(size);
    for _ in 0..size {
        random_values1.push(Fp::rand(&mut rng));
        random_values2.push(Fp::rand(&mut rng));
        random_values1.push(Fp::rand(&mut rng));
        random_values2.push(Fp::rand(&mut rng));
        random_values1.push(Fp::rand(&mut rng));
        random_values2.push(Fp::rand(&mut rng));
        random_values1.push(Fp::rand(&mut rng));
        random_values2.push(Fp::rand(&mut rng));
    }
    (random_values1, random_values2)
}

fn bench_ark_shared(c: &mut Criterion, random_values1: &Vec<Fp>, random_values2: &Vec<Fp>) {
    c.bench_function("ark multiplication", |b| {
        b.iter(|| {
            let mut results = [Fp::one(); 4];

            for i in (0..random_values1.len()).step_by(4) {
                results[0] *= random_values1[i] * random_values2[i];
                results[1] *= random_values1[i + 1] * random_values2[i + 1];
                results[2] *= random_values1[i + 2] * random_values2[i + 2];
                results[3] *= random_values1[i + 3] * random_values2[i + 3];
            }
            black_box(results)
        });
    });
}

fn bench_vector_shared(c: &mut Criterion, random_values1: &Vec<Fp>, random_values2: &Vec<Fp>) {
    c.bench_function("AVX multiplication", |b| {
        b.iter(|| {
            let mut results = pack4fields(Fp::one(), Fp::one(), Fp::one(), Fp::one());

            for i in (0..random_values1.len()).step_by(4) {
                unsafe {
                    let prefetch_index = i + 16;
                    if prefetch_index < random_values1.len() {
                        _mm_prefetch(random_values1.as_ptr().add(prefetch_index) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(random_values2.as_ptr().add(prefetch_index) as *const i8, _MM_HINT_T0);
                    }
                }

                let a = pack4fields(
                    random_values1[i],
                    random_values1[i + 1],
                    random_values1[i + 2],
                    random_values1[i + 3],
                );
                let b = pack4fields(
                    random_values2[i],
                    random_values2[i + 1],
                    random_values2[i + 2],
                    random_values2[i + 3],
                );
                mul_assign_fp4(&mut results, &mul_fp4(&a, &b));
            }

            black_box(results)
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    // Generate the random vectors only once.
    let (random_values1, random_values2) = generate_random_values();
    bench_ark_shared(c, &random_values1, &random_values2);
    bench_vector_shared(c, &random_values1, &random_values2);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
