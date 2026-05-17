/// Build script: compiles metal_wrapper.c and links Metal.framework.
fn main() {
    // Get macOS SDK path
    let sdk_path = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    println!("cargo:rerun-if-changed=src/metal_wrapper.m");

    // Compile the ObjC wrapper (v2: multi-expert support)
    cc::Build::new()
        .file("src/metal_wrapper.m")
        .flag(&format!("-isysroot{}", sdk_path))
        .compile("metal_wrapper");

    // Link frameworks
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
