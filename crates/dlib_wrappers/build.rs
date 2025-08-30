extern crate cpp_build;

fn main() {
    println!("cargo:rerun-if-changed=files");
    println!("cargo:rerun-if-changed=src");

    println!("cargo:rustc-link-search=/usr/lib");
    println!("cargo:rustc-link-search=/usr/include");

    let has_dlib = pkg_config::Config::new().probe("dlib").is_ok();
    if !has_dlib {
        println!("cargo:rustc-link-lib=dylib=dlib");
        println!("cargo:rustc-link-search=native=/usr/lib");
    }

    if pkg_config::Config::new().probe("openblas").is_err() {
        if pkg_config::Config::new().probe("cblas").is_err() {
            if pkg_config::Config::new().probe("blas").is_err() {
                // Final fallback: most systems with OpenBLAS expose it as -lopenblas
                println!("cargo:rustc-link-lib=dylib=openblas");
            }
        }
    }

    if pkg_config::Config::new().probe("lapack").is_err() {
        // Some distros bundle LAPACK into OpenBLAS; if a separate lapack isn't found
        println!("cargo:rustc-link-lib=dylib=lapack");
    }

    cpp_build::Config::new()
        .include("/usr/include")
        .include("/usr/local/include")
        .flag("-std=c++14")
        .build("src/lib.rs");
}
