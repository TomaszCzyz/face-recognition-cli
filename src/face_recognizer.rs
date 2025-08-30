use crate::image_helpers::{draw_point, draw_rectangle};
use crate::otel::{HISTOGRAM_F_D, HISTOGRAM_F_E, HISTOGRAM_L_P};
use crate::person_registry::person_registry::PersonRegistry;
use crate::{DefaultModels, get_output_path};
use dlib_wrappers::ImageMatrix;
use dlib_wrappers::face_detection::FaceDetectorModel;
use image::{Rgb, RgbImage, open};
use log::info;
use opentelemetry::KeyValue;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io};
use uuid::Uuid;

pub struct FaceRecognizer {
    models: DefaultModels,
    person_registry: Box<dyn PersonRegistry>,
}

impl FaceRecognizer {
    pub fn new(
        default_models: DefaultModels,
        person_registry: impl PersonRegistry + 'static,
    ) -> Self {
        Self {
            models: default_models,
            person_registry: Box::new(person_registry),
        }
    }

    pub fn process_path(&mut self, input: &PathBuf) {
        if input.is_dir() {
            let _ = visit_dirs(input, &mut |dir_entry| {
                let path = &dir_entry.path();
                self.process_file(&path);
            });
        } else if input.is_file() {
            self.process_file(&input);
        }
    }

    pub fn process_file(&mut self, input: &PathBuf) {
        info!("processing: {:?}", input);
        let mut image = open(input).unwrap().to_rgb8();

        self.detect_and_mark(&mut image);

        let output = get_output_path(input);
        image.save(&output).unwrap();
    }

    fn detect_and_mark(&mut self, image: &mut RgbImage) {
        let start = Instant::now();
        println!("starting...");

        let color = Rgb([255, 0, 0]);
        let matrix = ImageMatrix::from_image(&image);

        let face_locations_start = Instant::now();
        let face_locations = self.models.face_detector.face_locations(&matrix);
        println!(
            "found {:?} faces in {:?}",
            face_locations.len(),
            face_locations_start.elapsed()
        );

        HISTOGRAM_F_D.record(
            face_locations_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        for r in face_locations.iter() {
            draw_rectangle(image, &r, color);

            let landmarks_start = Instant::now();
            let landmarks = self.models.landmarks_predictor.face_landmarks(&matrix, &r);
            println!("finding landmarks took: {:?}", landmarks_start.elapsed());
            HISTOGRAM_L_P.record(
                landmarks_start.elapsed().as_millis() as u64,
                &[KeyValue::new("file", "file_name")],
            );

            let face_encoding_start = Instant::now();
            let encodings =
                self.models
                    .face_encoding
                    .get_face_encodings(&matrix, &[landmarks.clone()], 0);
            println!(
                "calculating encodings took: {:?}",
                face_encoding_start.elapsed()
            );
            let id = Uuid::new_v4();
            HISTOGRAM_F_E.record(
                face_encoding_start.elapsed().as_millis() as u64,
                &[
                    KeyValue::new("file", "file_name"),
                    KeyValue::new("id", id.to_string()),
                ],
            );

            for encoding in encodings.iter() {
                if let Some(name) = self.person_registry.get(encoding) {
                    println!("found person in registry: {}", name);
                } else {
                    let id = Uuid::new_v4();
                    self.person_registry.add(encoding.clone(), id.to_string());
                }
            }

            for point in landmarks.iter() {
                draw_point(image, &point, color);
            }
        }

        println!("finished {:?}", start.elapsed());
    }
}

// one possible implementation of walking a directory only visiting files
fn visit_dirs(dir: &Path, cb: &mut dyn FnMut(&DirEntry)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(&entry);
            }
        }
    }
    Ok(())
}
