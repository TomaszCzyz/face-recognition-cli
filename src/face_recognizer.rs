use crate::otel::{HISTOGRAM_F_D, HISTOGRAM_F_E, HISTOGRAM_L_P};
use crate::person_registry::person_registry::PersonRegistry;
use crate::person_registry::person_registry_sqlite::{PersonRegistrySqlite, ProcessedFileInsert};
use crate::{DefaultModels, get_output_path};
use dlib_wrappers::face_detection::{FaceDetectorModel, FaceLocations};
use dlib_wrappers::face_encoding::FaceEncodings;
use dlib_wrappers::landmark_prediction::FaceLandmarks;
use dlib_wrappers::{ImageMatrix, Rectangle};
use image::{RgbImage, open};
use opentelemetry::KeyValue;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

pub struct FaceRecognizer {
    models: DefaultModels,
    person_registry: PersonRegistrySqlite,
}

impl Debug for FaceRecognizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaceRecognizer")
    }
}

impl FaceRecognizer {
    pub fn new(default_models: DefaultModels, person_registry: PersonRegistrySqlite) -> Self {
        Self {
            models: default_models,
            person_registry,
        }
    }

    #[instrument(skip(self), name = "processing file")]
    pub async fn process_file(&mut self, input: &Path) {
        let mut image = open(input).unwrap().to_rgb8();

        let hash = blake3::hash(image.as_raw());

        match self.person_registry.find_file(&hash).await {
            Some((_id, path, processed_at)) => {
                info!(
                    "the file has already been processed at {}, its path was: {}",
                    processed_at, path
                );
                return;
            }
            None => {
                self.person_registry
                    .add_file(ProcessedFileInsert {
                        hash,
                        path: input.to_string_lossy().to_string(),
                    })
                    .await
            }
        };

        self.detect(&mut image);

        let output = get_output_path(input);
        image.save(&output).unwrap();
    }

    #[instrument(skip(self, image), name = "detecting faces")]
    fn detect(&mut self, image: &mut RgbImage) {
        let start = Instant::now();
        let matrix = ImageMatrix::from_image(&image);

        let face_locations = self.find_face_locations(&matrix);

        // todo: save face locations

        for r in face_locations.iter() {
            let landmarks = self.find_landmarks(&matrix, &r);
            let encodings = self.calculate_face_encoding(&matrix, &landmarks);

            for encoding in encodings.iter() {
                if let Some(name) = self.person_registry.get(encoding) {
                    debug!("found person in registry: {}", name);
                } else {
                    let id = Uuid::new_v4();
                    // self.person_registry.add(encoding.clone(), id.to_string());
                }
            }
        }

        info!("finished {:?}", start.elapsed());
    }

    fn calculate_face_encoding(
        &mut self,
        matrix: &ImageMatrix,
        landmarks: &FaceLandmarks,
    ) -> FaceEncodings {
        let face_encoding_start = Instant::now();
        let encodings =
            self.models
                .face_encoding
                .get_face_encodings(&matrix, &[landmarks.clone()], 0);

        HISTOGRAM_F_E.record(
            face_encoding_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        debug!(
            "calculating encodings took: {:?}",
            face_encoding_start.elapsed()
        );

        encodings
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

    fn find_face_locations(&mut self, matrix: &ImageMatrix) -> FaceLocations {
        let face_locations_start = Instant::now();
        let face_locations = self.models.face_detector.face_locations(&matrix);

        HISTOGRAM_F_D.record(
            face_locations_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        info!(
            "found {:?} faces in {:?}",
            face_locations.len(),
            face_locations_start.elapsed()
        );

        face_locations
    }
}
