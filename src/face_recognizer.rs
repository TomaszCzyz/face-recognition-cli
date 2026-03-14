use crate::DefaultModels;
use crate::otel::{HISTOGRAM_F_D, HISTOGRAM_F_E, HISTOGRAM_L_P};
use crate::person_registry::person_registry_sqlite::{PersonRegistrySqlite, ProcessedFileInsert};
use blake3::Hash;
use dlib_wrappers::face_detection::{FaceDetectorModel, FaceLocations};
use dlib_wrappers::face_encoding::FaceEncodings;
use dlib_wrappers::landmark_prediction::FaceLandmarks;
use dlib_wrappers::{ImageMatrix, Rectangle};
use image::{RgbImage, open};
use memmap2::Mmap;
use opentelemetry::KeyValue;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs;
use tracing::{info, instrument};

pub struct FaceRecognizer {
    models: Arc<DefaultModels>,
    person_registry: PersonRegistrySqlite,
}

#[derive(Copy, Clone, Debug)]
pub struct FaceRecognizerOptions {
    pub(crate) skip_processed_check: bool,
}

impl Debug for FaceRecognizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaceRecognizer")
    }
}

impl FaceRecognizer {
    pub fn new(models: Arc<DefaultModels>, person_registry: PersonRegistrySqlite) -> Self {
        Self {
            models,
            person_registry,
        }
    }

    #[instrument(skip(self, options), name = "processing file")]
    pub async fn process_file(&self, input: &Path, options: FaceRecognizerOptions) {
        let hash = Self::calc_hash(input);

        let mut is_processed = false;

        let file_id = match self.person_registry.find_file(&hash).await {
            Some((id, path, processed_at)) => {
                // todo: update file path if different
                info!(
                    "the file has already been processed at {}, its path was: {}",
                    processed_at, path
                );
                is_processed = true;
                id
            }
            None => {
                let canonical_path = fs::canonicalize(&input)
                    .await
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                self.person_registry
                    .add_file(ProcessedFileInsert {
                        hash,
                        path: canonical_path,
                    })
                    .await
            }
        };

        if is_processed && !options.skip_processed_check {
            return;
        }

        let mut image = open(input).unwrap().to_rgb8();
        self.detect(&mut image, file_id).await;
    }

    #[instrument(skip(self, image), name = "detecting faces")]
    async fn detect(&self, image: &mut RgbImage, file_id: i64) {
        let start = Instant::now();
        let matrix = ImageMatrix::from_image(&image);

        let face_locations = self.find_face_locations(&matrix);
        let faces_landmarks = self.find_landmarks(&matrix, &face_locations);
        let encodings = self.calculate_face_encodings(&matrix, faces_landmarks.as_slice());

        for (location_rect, encoding) in face_locations.iter().zip(encodings.iter()) {
            _ = self
                .person_registry
                .add_face(Some(file_id), encoding, location_rect)
                .await;
        }

        info!("finished {:?}", start.elapsed());
    }

    fn calc_hash(input: &Path) -> Hash {
        let file = File::open(input).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        blake3::hash(&mmap)
    }

    fn find_landmarks(&self, matrix: &ImageMatrix, rectangles: &[Rectangle]) -> Vec<FaceLandmarks> {
        let landmarks_start = Instant::now();

        let all_landmarks = rectangles
            .iter()
            .map(|face_location_rect| {
                self.models
                    .landmarks_predictor
                    .face_landmarks(&matrix, &face_location_rect)
            })
            .collect();

        HISTOGRAM_L_P.record(
            landmarks_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        info!("finding landmarks took: {:?}", landmarks_start.elapsed());

        all_landmarks
    }

    fn find_face_locations(&self, matrix: &ImageMatrix) -> FaceLocations {
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

    fn calculate_face_encodings(
        &self,
        matrix: &ImageMatrix,
        landmarks: &[FaceLandmarks],
    ) -> FaceEncodings {
        let face_encoding_start = Instant::now();
        let encodings = self
            .models
            .face_encoding
            .get_face_encodings(&matrix, landmarks, 0);

        HISTOGRAM_F_E.record(
            face_encoding_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        info!(
            "calculating encodings took: {:?}",
            face_encoding_start.elapsed()
        );

        encodings
    }
}
