use dlib_wrappers::face_encoding::FaceEncoding;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub struct FilePersonRegistry {
    storage_path: PathBuf,
    registry: HashMap<FaceEncoding, String>,

    // encoding -> name
    // name -> Vec<encoding>
}

impl FilePersonRegistry {
    pub(crate) fn new(path: &PathBuf) -> Self {
        if path.exists() {
            Self::read_from_file(path)
        } else {
            Self {
                storage_path: PathBuf::from(path),
                registry: HashMap::new(),
            }
        }
    }

    fn read_from_file(path: &Path) -> Self {
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("Failed to read registry file: {}", e);
        });

        let mut registry = HashMap::new();

        for line in content.lines() {
            if line.is_empty() {
                continue;
            }

            let mut parts = line.split('|');
            let encoding_str = parts.next().unwrap();
            let name = parts.next().unwrap().to_string();

            let encoding_str = encoding_str.trim_start_matches('[').trim_end_matches(']');

            let values: Vec<f64> = encoding_str
                .split(',')
                .map(|s| f64::from_str(s.trim()).unwrap())
                .collect();

            let encoding = FaceEncoding::new(values);
            registry.insert(encoding, name);
        }

        Self {
            storage_path: PathBuf::from(path),
            registry,
        }
    }

    pub fn get(&self, encoding: &FaceEncoding) -> Option<&String> {
        for (key, value) in &self.registry {
            let x = key.distance(encoding);
            println!("dist: {:?}", x);
            if x < 0.6 {
                return Some(value);
            }
        }

        None
    }

    pub fn add(&mut self, encoding: FaceEncoding, name: String) {
        // TODO: what about different encodings for the same name
        self.registry.entry(encoding).or_insert(name);
    }

    pub fn save(&self) {
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!("Failed to create directory structure: {}", e);
            });
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.storage_path)
            .unwrap();

        let mut writer = BufWriter::new(file);

        for (encoding, name) in &self.registry {
            writeln!(writer, "{:?}|{}", encoding, name).unwrap();
        }
    }
}
