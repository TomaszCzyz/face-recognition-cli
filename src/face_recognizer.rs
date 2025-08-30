use crate::image_helpers::{draw_point, draw_rectangle};
use crate::otel::{HISTOGRAM_F_D, HISTOGRAM_F_E, HISTOGRAM_L_P};
use crate::person_registry::person_registry::PersonRegistry;
use crate::{DefaultModels, get_output_path};
use dlib_wrappers::face_detection::FaceDetectorModel;
use dlib_wrappers::landmark_prediction::FaceLandmarks;
use dlib_wrappers::{ImageMatrix, Rectangle};
use image::{Rgb, RgbImage, open};
use opentelemetry::KeyValue;
use std::fmt::{Debug, Formatter};
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use tracing::{debug, info, instrument};
use uuid::Uuid;

pub struct FaceRecognizer {
    models: DefaultModels,
    person_registry: Box<dyn PersonRegistry>,
}

impl Debug for FaceRecognizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaceRecognizer")
    }
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

    #[instrument(skip(self), name = "processing file")]
    pub fn process_file(&mut self, input: &PathBuf) {
        let mut image = open(input).unwrap().to_rgb8();

        self.detect(&mut image);

        let output = get_output_path(input);
        image.save(&output).unwrap();
    }

    #[instrument(skip(self, image), name = "detecting faces")]
    fn detect(&mut self, image: &mut RgbImage) {
        let start = Instant::now();
        let matrix = ImageMatrix::from_image(&image);

        let face_locations_start = Instant::now();
        let face_locations = self.models.face_detector.face_locations(&matrix);
        info!(
            "found {:?} faces in {:?}",
            face_locations.len(),
            face_locations_start.elapsed()
        );

        // todo: save face locations

        HISTOGRAM_F_D.record(
            face_locations_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        for r in face_locations.iter() {
            let landmarks = self.find_landmarks(&matrix, &r);

            let face_encoding_start = Instant::now();
            let encodings =
                self.models
                    .face_encoding
                    .get_face_encodings(&matrix, &[landmarks.clone()], 0);
            debug!(
                "calculating encodings took: {:?}",
                face_encoding_start.elapsed()
            );

            HISTOGRAM_F_E.record(
                face_encoding_start.elapsed().as_millis() as u64,
                &[KeyValue::new("file", "file_name")],
            );

            for encoding in encodings.iter() {
                if let Some(name) = self.person_registry.get(encoding) {
                    debug!("found person in registry: {}", name);
                } else {
                    let id = Uuid::new_v4();
                    self.person_registry.add(encoding.clone(), id.to_string());
                }
            }
        }

        info!("finished {:?}", start.elapsed());
    }

    fn find_landmarks(&mut self, matrix: &ImageMatrix, r: &Rectangle) -> FaceLandmarks {
        let landmarks_start = Instant::now();
        let landmarks = self.models.landmarks_predictor.face_landmarks(&matrix, &r);

        HISTOGRAM_L_P.record(
            landmarks_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        landmarks
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
