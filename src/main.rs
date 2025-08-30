#![allow(dead_code)]
use crate::face_recognizer::FaceRecognizer;
use crate::person_registry::person_registry_sqlite::PersonRegistrySqlite;
use directories::ProjectDirs;
use dlib_wrappers::face_detection::{FaceDetectorCnn};
use dlib_wrappers::face_encoding::FaceEncodingNetwork;
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use once_cell::sync::Lazy;
use std::path::PathBuf;

mod face_recognizer;
mod image_helpers;
mod otel;
mod person_registry;

static PROJECT_DIRS: Lazy<ProjectDirs> = Lazy::new(|| {
    ProjectDirs::from("com", "example", "face-recognizer")
        .expect("failed to determine project directories")
});

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // force early init of the project dirs to handle panic
    let _ = PROJECT_DIRS.data_dir();

    // let provider = init_metrics();

    let models = DefaultModels::default();

    let persons_registry = PersonRegistrySqlite::initialize().await;

    let mut recognizer = FaceRecognizer::new(models, persons_registry);

    let cmd = clap::Command::new("face-recognizer")
        .subcommand_required(true)
        .subcommand(
            clap::command!("recognize").args(&[
                clap::arg!(<input> "a path to a file to analyse")
                    .value_parser(clap::value_parser!(PathBuf)),
                clap::arg!(--"names-path" <PATH>).value_parser(clap::value_parser!(PathBuf)),
            ]),
        );

    let matches = cmd.get_matches();
    match matches.subcommand() {
        Some(("recognize", matches)) => {
            let input = matches.get_one::<PathBuf>("input").unwrap();
            recognizer.process_file(&input);


            if input.is_dir() {
                for dir_entry in std::fs::read_dir(input)
                    .unwrap()
                    .filter(|x| x.as_ref().is_ok_and(|x| x.path().is_file()))
                {
                    let path = &dir_entry.unwrap().path();
                    recognizer.process_file(&path);
                }
            } else if input.is_file() {
                recognizer.process_file(&input);
            }
        }
        _ => unreachable!("clap should ensure we don't get here"),
    };

    // provider.shutdown()?;

    Ok(())
}

fn get_output_path(input: &PathBuf) -> PathBuf {
    let input_filename = input.file_stem().unwrap().to_str().unwrap();
    let input_ext = input.extension().unwrap().to_str().unwrap();
    let output = input
        .parent()
        .unwrap()
        .join(format!("{}_new.{}", input_filename, input_ext));

    output
}

struct DefaultModels {
    face_detector: FaceDetectorCnn,
    landmarks_predictor: LandmarkPredictor,
    face_encoding: FaceEncodingNetwork,
}

impl Default for DefaultModels {
    fn default() -> Self {
        Self {
            // detector: FaceDetector::default(),
            face_detector: FaceDetectorCnn::new(
                "crates/dlib_wrappers/files/mmod_human_face_detector.dat",
            )
            .expect("Failed to load CNN face detector"),
            landmarks_predictor: LandmarkPredictor::new(
                "crates/dlib_wrappers/files/shape_predictor_68_face_landmarks.dat",
            )
            .expect("Failed to load landmark predictor"),
            face_encoding: FaceEncodingNetwork::new(
                "crates/dlib_wrappers/files/dlib_face_recognition_resnet_model_v1.dat",
            )
            .expect("Failed to load face encoding network"),
        }
    }
}
