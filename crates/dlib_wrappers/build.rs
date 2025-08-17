extern crate cpp_build;

fn main() {
    println!("cargo:rerun-if-changed=files");
    println!("cargo:rerun-if-changed=src");

    println!("cargo:rustc-link-search=/usr/lib");
    println!("cargo:rustc-link-search=/usr/include");

    println!("cargo:rustc-link-lib=dlib");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=cblas");

    cpp_build::Config::new()
        .include("/usr/include")
        .include("/usr/local/include")
        .flag("-std=c++14")
        .build("src/lib.rs");
}
